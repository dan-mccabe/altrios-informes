"""Proof-of-concept: run LIFTS for all terminals using an externally-generated
train schedule instead of sim_manager.main().

Schedule convention (from INFORMES):
  Each row represents a dwell at ONE endpoint of the route.
  - route=XXXXX--TERMINAL, direction=forward  → train is at TERMINAL (arrived from XXXXX)
  - route=TERMINAL--XXXXX, direction=reverse  → train is at TERMINAL (returned from XXXXX)
"""
from pathlib import Path

import polars as pl

from altrios.lifts import TerminalMode, register_mode, run_terminal_simulation
from altrios.lifts.terminal_sim import loggingLevel
from altrios.lifts.train_flow import process_train_arrival

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CARS_PER_TRAIN = 120   # placeholder — replace with per-trip values from routing

SCHEDULE_PATH = Path(__file__).parent / "train_schedule.csv"

# ---------------------------------------------------------------------------
# Read schedule and extract all unique terminal IDs
# ---------------------------------------------------------------------------
df = pl.read_csv(SCHEDULE_PATH)

all_terminals = sorted(
    pl.concat([
        df.select(pl.col("route").str.split("--").list.get(0).alias("terminal")),
        df.select(pl.col("route").str.split("--").list.get(1).alias("terminal")),
    ])
    .unique()["terminal"]
    .to_list()
)
print(f"Terminals found: {all_terminals}")

# ---------------------------------------------------------------------------
# Register a passthrough mode so LIFTS accepts our pre-built timetable.
# Done once before the loop.
# ---------------------------------------------------------------------------
def _build_passthrough(timetable_dicts, terminal):
    """Schedule is already built; just return it."""
    return timetable_dicts


register_mode(TerminalMode(
    name="external_schedule",
    process_arrival=process_train_arrival,
    build_schedule=_build_passthrough,
    description="Pre-built timetable from an external scheduler (e.g. INFORMES).",
))

# ---------------------------------------------------------------------------
# Run LIFTS for each terminal and collect transfer times
# ---------------------------------------------------------------------------
def compute_transfer_times(container_data: pl.DataFrame, terminal: str) -> pl.DataFrame:
    """Extract per-container transfer times from a LIFTS container_data result.

    IC: hostler_dropoff - train_arrival_actual
    OC: crane_load - max(crane_unload) across all ICs for the same train
    """
    df = container_data.with_columns(
        pl.col("container_id").str.extract(r"Train-(\d+)").cast(pl.Int64).alias("train_id"),
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


all_transfer_times = []

for terminal in all_terminals:
    at_terminal = (
        df.filter(
            (pl.col("route").str.ends_with(f"--{terminal}") & (pl.col("direction") == "forward"))
            | (pl.col("route").str.starts_with(f"{terminal}--") & (pl.col("direction") == "reverse"))
        )
        .sort("arrival_time")
    )

    if at_terminal.height == 0:
        continue

    timetable = [
        {
            "train_id":       int(row["vehicle_id"]) * 1000 + int(row["trip_idx"]),
            "arrival_time":   row["arrival_time"],
            "departure_time": row["departure_time"],
            "full_cars":      CARS_PER_TRAIN,
            "empty_cars":     0,
            "oc_number":      CARS_PER_TRAIN,
            "truck_number":   CARS_PER_TRAIN,
        }
        for row in at_terminal.iter_rows(named=True)
    ]

    print(f"\n{'='*60}")
    print(f"Terminal {terminal}: {len(timetable)} train visits")

    container_data, _, _ = run_terminal_simulation(
        mode="external_schedule",
        train_consist_plan=timetable,
        terminal=terminal,
        log_level=loggingLevel.NONE,
    )

    transfer_times = compute_transfer_times(container_data, terminal)
    all_transfer_times.append(transfer_times)
    print(f"  IC rows: {transfer_times.filter(pl.col('direction')=='inbound').height}")
    print(f"  OC rows: {transfer_times.filter(pl.col('direction')=='outbound').height}")

# ---------------------------------------------------------------------------
# Combine and save
# ---------------------------------------------------------------------------
combined = (
    pl.concat(all_transfer_times)
    .sort("location", "vehicle_id", "trip_idx", "direction")
)

combined.write_csv("transfer_times.csv")
print(f"\nTotal rows written: {combined.height}")

# ---------------------------------------------------------------------------
# Verify all trips are represented
# ---------------------------------------------------------------------------
# Every row in the schedule is one trip at one terminal.
# Each trip should produce CARS_PER_TRAIN inbound + CARS_PER_TRAIN outbound rows.
schedule_trips = df.select("vehicle_id", "trip_idx", pl.col("route"), pl.col("direction")).with_columns(
    pl.when(pl.col("direction") == "forward")
        .then(pl.col("route").str.split("--").list.get(1))
        .otherwise(pl.col("route").str.split("--").list.get(0))
        .alias("location")
).select("location", "vehicle_id", "trip_idx")

combined_trips = (
    combined
    .group_by("location", "vehicle_id", "trip_idx")
    .agg(pl.len().alias("container_count"))
)

missing = schedule_trips.join(combined_trips, on=["location", "vehicle_id", "trip_idx"], how="anti")
if missing.height == 0:
    print(f"Verification passed: all {schedule_trips.height} trips represented in transfer_times.csv")
else:
    print(f"WARNING: {missing.height} trips missing from output:")
    print(missing)
