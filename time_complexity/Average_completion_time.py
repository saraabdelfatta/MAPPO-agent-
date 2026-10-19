"""Calculate mean task completion time relative to mission start."""

import argparse
import math
from collections.abc import Sequence


def calculate_average_completion_time(
    start_time_s: float, completion_times_s: Sequence[float]
) -> float:
    """Return the mean completion offset from mission start."""
    if not math.isfinite(start_time_s):
        raise ValueError("start_time_s must be finite")
    if not completion_times_s:
        raise ValueError("at least one task completion time is required")
    if any(not math.isfinite(value) for value in completion_times_s):
        raise ValueError("completion times must be finite")
    if any(value < start_time_s for value in completion_times_s):
        raise ValueError("task completion cannot be before mission start")
    return sum(value - start_time_s for value in completion_times_s) / len(
        completion_times_s
    )


def main() -> None:
    """Parse timestamps and print average completion time."""
    parser = argparse.ArgumentParser(
        description="Calculate the mean task completion time from simulator timestamps."
    )
    parser.add_argument("--start", type=float, required=True, help="Mission start time in sim seconds.")
    parser.add_argument(
        "--completion-times",
        type=float,
        nargs="+",
        required=True,
        help="Absolute simulator completion timestamps for tasks or robot finishes.",
    )
    args = parser.parse_args()
    average = calculate_average_completion_time(args.start, args.completion_times)
    print(f"Average completion time: {average:.3f} sim seconds")


if __name__ == "__main__":
    main()
