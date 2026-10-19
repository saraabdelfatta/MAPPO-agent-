import unittest

from evaluation.timing import MissionTimingEvaluator


class MissionTimingTests(unittest.TestCase):
    def setUp(self):
        self.metrics = MissionTimingEvaluator(
            mission_id="scenario-01-seed-3",
            controller="mappo",
            start_sim_time_s=10.0,
            required_task_ids=frozenset({"task-a", "task-b"}),
            required_robot_ids=frozenset({"robot-a", "robot-b"}),
        )

    def test_success_reports_mission_task_robot_and_inference_times(self):
        self.metrics.record_task_completion("task-a", 13.0)
        self.metrics.record_task_completion("task-b", 18.0)
        self.metrics.record_robot_finish("robot-a", 14.0)
        self.metrics.record_robot_finish("robot-b", 20.0)
        for duration in (2.0, 4.0, 8.0):
            self.metrics.record_inference_time(duration)

        result = self.metrics.result(end_sim_time_s=20.0, status="success")

        self.assertTrue(result["success"])
        self.assertEqual(result["makespan_sim_s"], 8.0)
        self.assertEqual(result["mean_task_completion_sim_s"], 5.5)
        self.assertEqual(result["mean_robot_finish_sim_s"], 7.0)
        self.assertEqual(result["termination_elapsed_sim_s"], 10.0)
        self.assertEqual(result["mean_inference_ms"], 14.0 / 3.0)
        self.assertEqual(result["p50_inference_ms"], 4.0)
        self.assertEqual(result["p95_inference_ms"], 7.6)
        self.assertEqual(result["p99_inference_ms"], 7.92)

    def test_timeout_does_not_report_successful_mission_times(self):
        self.metrics.record_task_completion("task-a", 13.0)

        result = self.metrics.result(end_sim_time_s=20.0, status="timeout")

        self.assertFalse(result["success"])
        self.assertIsNone(result["makespan_sim_s"])
        self.assertIsNone(result["mean_task_completion_sim_s"])
        self.assertEqual(result["termination_elapsed_sim_s"], 10.0)

    def test_success_requires_every_required_task(self):
        with self.assertRaisesRegex(ValueError, "every required task"):
            self.metrics.result(end_sim_time_s=20.0, status="success")

    def test_rejects_unknown_task_and_event_before_start(self):
        with self.assertRaisesRegex(ValueError, "unexpected task_id"):
            self.metrics.record_task_completion("unknown", 11.0)
        with self.assertRaisesRegex(ValueError, "before mission start"):
            self.metrics.record_task_completion("task-a", 9.0)

    def test_keeps_first_task_completion_timestamp(self):
        self.metrics.record_task_completion("task-a", 12.0)
        self.metrics.record_task_completion("task-a", 15.0)

        self.assertEqual(self.metrics.task_completion_sim_times_s["task-a"], 12.0)

    def test_rejects_non_finite_or_negative_inference_duration(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            self.metrics.record_inference_time(float("nan"))
        with self.assertRaisesRegex(ValueError, "cannot be negative"):
            self.metrics.record_inference_time(-1.0)


if __name__ == "__main__":
    unittest.main()
