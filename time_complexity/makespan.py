"""Calculate mission makespan from task completion timestamps."""

import argparse
import math
from collections.abc import Sequence


def calculate_makespan(start_time_s: float, completion_times_s: Sequence[float]) -> float:
    """Return the last completion offset from mission start."""
    if not math.isfinite(start_time_s):
        raise ValueError("start_time_s must be finite")
    if not completion_times_s:
        raise ValueError("at least one task completion time is required")
    if any(not math.isfinite(value) for value in completion_times_s):
        raise ValueError("completion times must be finite")
    if any(value < start_time_s for value in completion_times_s):
        raise ValueError("task completion cannot be before mission start")
    return max(completion_times_s) - start_time_s


def main() -> None:
    """Parse timestamps and print the mission makespan."""
    parser = argparse.ArgumentParser(
        description="Calculate mission makespan from task completion times."
    )
    parser.add_argument("--start", type=float, required=True, help="Mission start time in sim seconds.")
    parser.add_argument(
        "--completion-times",
        type=float,
        nargs="+",
        required=True,
        help="Absolute simulator completion timestamps for required tasks.",
    )
    args = parser.parse_args()
    print(
        f"Mission makespan: "
        f"{calculate_makespan(args.start, args.completion_times):.3f} sim seconds"
    )


if __name__ == "__main__":
    main()
