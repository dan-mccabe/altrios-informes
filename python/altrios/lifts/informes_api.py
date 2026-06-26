"""LIFTS driver for externally-generated (e.g. INFORMES) train schedules.

Public entry point
------------------
    get_container_transfer_times(schedule, *, cars_per_train, log_level)

The function runs LIFTS for every terminal found in *schedule* and returns a
single DataFrame of per-container transfer times — no file I/O, no global
side-effects beyond the one-time mode registration that happens at import.
"""
from __future__ import annotations

from typing import Union

import pandas as pd
import polars as pl

from altrios.lifts.classes import loggingLevel
from altrios.lifts.terminal_sim import (
    TerminalMode,
    register_mode,
    run_terminal_simulation,
)
from altrios.lifts.train_flow import process_train_arrival

# ---------------------------------------------------------------------------
# Passthrough mode — registered once at import time
# ---------------------------------------------------------------------------
_MODE_NAME = "external_schedule"


def _passthrough_build(timetable_dicts, terminal):
    """Schedule is pre-built; return it unchanged."""
    return timetable_dicts


try:
    register_mode(
        TerminalMode(
            name=_MODE_NAME,
            process_arrival=process_train_arrival,
            build_schedule=_passthrough_build,
            description=(
                "Pre-built timetable from an external scheduler (e.g. INFORMES). "
                "Registered automatically on import of altrios.lifts.informes_api."
            ),
        )
    )
except ValueError:
    pass  # already registered from a previous import


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_container_transfer_times(
    schedule: Union[pd.DataFrame, pl.DataFrame],
    cars_per_train: int = 120,
    log_level: loggingLevel = loggingLevel.NONE,
) -> Union[pd.DataFrame, pl.DataFrame]:
    """Run LIFTS for all terminals in *schedule* and return transfer times.

    Parameters
    ----------
    schedule:
        pandas or polars DataFrame with columns:
          ``route`` (str, ``"A--B"``), ``direction`` (``"forward"`` | ``"reverse"``),
          ``vehicle_id`` (int), ``trip_idx`` (int),
          ``arrival_time`` (float, hours), ``departure_time`` (float, hours).

        Optionally a ``cars_per_train`` (int) column may be present; when it
        is, its per-row value overrides the scalar *cars_per_train* argument.

        Schedule convention (from INFORMES):
          Each row is a dwell at ONE endpoint of the route.

          * ``route=XXXXX--TERMINAL``, ``direction=forward``  → train is **at TERMINAL**
          * ``route=TERMINAL--XXXXX``, ``direction=reverse``  → train is **at TERMINAL**

    cars_per_train:
        Number of cars (ICs / OCs / trucks) per train visit when the column is
        absent from *schedule*.
    log_level:
        LIFTS verbosity forwarded to each terminal simulation.

    Returns
    -------
    pandas or polars DataFrame (matches the type of *schedule*)
        One row per container with columns:
        ``location``, ``vehicle_id`` (int), ``trip_idx`` (int),
        ``direction`` (``"inbound"`` | ``"outbound"``),
        ``transfer_time`` (float, hours).

        Inbound (IC) transfer time  = ``hostler_dropoff`` - ``train_arrival_actual``
        Outbound (OC) transfer time = ``crane_load``      - last IC ``crane_unload``
    """
    return_pandas = isinstance(schedule, pd.DataFrame)
    if return_pandas:
        schedule = pl.from_pandas(schedule)

    has_cars_col = "cars_per_train" in schedule.columns

    all_terminals = sorted(
        pl.concat([
            schedule.select(pl.col("route").str.split("--").list.get(0).alias("t")),
            schedule.select(pl.col("route").str.split("--").list.get(1).alias("t")),
        ]).unique()["t"].to_list()
    )

    results: list[pl.DataFrame] = []

    for terminal in all_terminals:
        at_terminal = schedule.filter(
            (
                pl.col("route").str.ends_with(f"--{terminal}")
                & (pl.col("direction") == "forward")
            )
            | (
                pl.col("route").str.starts_with(f"{terminal}--")
                & (pl.col("direction") == "reverse")
            )
        ).sort("arrival_time")

        if at_terminal.is_empty():
            continue

        timetable = [
            {
                "train_id": int(r["vehicle_id"]) * 1000 + int(r["trip_idx"]),
                "arrival_time": r["arrival_time"],
                "departure_time": r["departure_time"],
                "full_cars": int(r["cars_per_train"]) if has_cars_col else cars_per_train,
                "empty_cars": 0,
                "oc_number": int(r["cars_per_train"]) if has_cars_col else cars_per_train,
                "truck_number": int(r["cars_per_train"]) if has_cars_col else cars_per_train,
            }
            for r in at_terminal.iter_rows(named=True)
        ]

        container_data, _, _ = run_terminal_simulation(
            mode=_MODE_NAME,
            train_consist_plan=timetable,
            terminal=terminal,
            log_level=log_level,
        )

        results.append(_compute_transfer_times(container_data, terminal))

    _EMPTY_SCHEMA = {
        "location": pl.Utf8,
        "vehicle_id": pl.Int64,
        "trip_idx": pl.Int64,
        "direction": pl.Utf8,
        "transfer_time": pl.Float64,
    }

    if not results:
        out = pl.DataFrame(schema=_EMPTY_SCHEMA)
    else:
        out = pl.concat(results).sort("location", "vehicle_id", "trip_idx", "direction")

    return out.to_pandas() if return_pandas else out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_transfer_times(container_data: pl.DataFrame, terminal: str) -> pl.DataFrame:
    """Derive IC and OC transfer times from a single terminal's container_data."""
    df = container_data.with_columns(
        pl.col("container_id")
        .str.extract(r"Train-(\d+)")
        .cast(pl.Int64)
        .alias("train_id"),
        pl.col("container_id").str.starts_with("IC").alias("is_ic"),
    )

    last_ic_crane_unload = (
        df.filter(pl.col("is_ic"))
        .group_by("train_id")
        .agg(pl.col("crane_unload").max().alias("last_ic_crane_unload"))
    )

    return (
        df.join(last_ic_crane_unload, on="train_id", how="left")
        .with_columns(
            pl.when(pl.col("is_ic"))
            .then(pl.col("hostler_dropoff") - pl.col("train_arrival_actual"))
            .otherwise(pl.col("crane_load") - pl.col("last_ic_crane_unload"))
            .alias("transfer_time")
        )
        .filter(pl.col("transfer_time").is_not_null())
        .with_columns(
            pl.lit(terminal).alias("location"),
            (pl.col("train_id") // 1000).alias("vehicle_id"),
            (pl.col("train_id") % 1000).alias("trip_idx"),
            pl.when(pl.col("is_ic"))
            .then(pl.lit("inbound"))
            .otherwise(pl.lit("outbound"))
            .alias("direction"),
        )
        .select("location", "vehicle_id", "trip_idx", "direction", "transfer_time")
    )
