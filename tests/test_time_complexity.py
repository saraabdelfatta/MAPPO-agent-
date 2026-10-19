import unittest

from time_complexity.Average_completion_time import calculate_average_completion_time
from time_complexity.makespan import calculate_makespan


class TimeMetricTests(unittest.TestCase):
    def test_makespan_uses_last_completion_relative_to_start(self):
        self.assertEqual(calculate_makespan(10.0, [22.0, 25.0, 28.0]), 18.0)

    def test_average_completion_time_uses_all_completion_offsets(self):
        self.assertEqual(
            calculate_average_completion_time(10.0, [22.0, 25.0, 28.0]), 15.0
        )

    def test_metrics_reject_missing_or_pre_start_completions(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            calculate_makespan(0.0, [])
        with self.assertRaisesRegex(ValueError, "before mission start"):
            calculate_average_completion_time(10.0, [9.0])


if __name__ == "__main__":
    unittest.main()
