"""Simulator-independent mission and policy timing metrics."""

from dataclasses import dataclass, field
from math import isfinite
from statistics import mean


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None

    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


@dataclass
class MissionTimingEvaluator:
    mission_id: str
    controller: str
    start_sim_time_s: float
    required_task_ids: frozenset[str]
    required_robot_ids: frozenset[str] = frozenset()
    task_completion_sim_times_s: dict[str, float] = field(default_factory=dict)
    robot_finish_sim_times_s: dict[str, float] = field(default_factory=dict)
    inference_times_ms: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.mission_id:
            raise ValueError("mission_id must not be empty")
        if not self.controller:
            raise ValueError("controller must not be empty")
        if not self.required_task_ids:
            raise ValueError("at least one required task must be specified")
        self._validate_time(self.start_sim_time_s, "start_sim_time_s")

    @staticmethod
    def _validate_time(value: float, name: str) -> None:
        if not isfinite(value):
            raise ValueError(f"{name} must be finite")

    def record_task_completion(self, task_id: str, sim_time_s: float) -> None:
        if task_id not in self.required_task_ids:
            raise ValueError(f"unexpected task_id: {task_id}")
        self._record_sim_time(
            self.task_completion_sim_times_s, task_id, sim_time_s
        )

    def record_robot_finish(self, robot_id: str, sim_time_s: float) -> None:
        if self.required_robot_ids and robot_id not in self.required_robot_ids:
            raise ValueError(f"unexpected robot_id: {robot_id}")
        self._record_sim_time(self.robot_finish_sim_times_s, robot_id, sim_time_s)

    def _record_sim_time(
        self, records: dict[str, float], item_id: str, sim_time_s: float
    ) -> None:
        self._validate_time(sim_time_s, "sim_time_s")
        if sim_time_s < self.start_sim_time_s:
            raise ValueError("event time cannot be before mission start")
        records.setdefault(item_id, sim_time_s)

    def record_inference_time(self, duration_ms: float) -> None:
        self._validate_time(duration_ms, "duration_ms")
        if duration_ms < 0:
            raise ValueError("duration_ms cannot be negative")
        self.inference_times_ms.append(duration_ms)

    def result(self, end_sim_time_s: float, status: str) -> dict[str, str | bool | int | float | None]:
        self._validate_time(end_sim_time_s, "end_sim_time_s")
        if end_sim_time_s < self.start_sim_time_s:
            raise ValueError("mission end cannot be before mission start")
        if not status:
            raise ValueError("status must not be empty")
        if any(time_s > end_sim_time_s for time_s in self.task_completion_sim_times_s.values()):
            raise ValueError("task completion cannot be after mission end")
        if any(time_s > end_sim_time_s for time_s in self.robot_finish_sim_times_s.values()):
            raise ValueError("robot finish cannot be after mission end")

        all_tasks_complete = self.required_task_ids.issubset(
            self.task_completion_sim_times_s
        )
        if status == "success" and not all_tasks_complete:
            raise ValueError("success status requires every required task to be complete")

        success = status == "success"
        task_elapsed_s = [
            self.task_completion_sim_times_s[task_id] - self.start_sim_time_s
            for task_id in self.required_task_ids
        ] if success else []
        all_robots_finished = bool(self.required_robot_ids) and (
            self.required_robot_ids.issubset(self.robot_finish_sim_times_s)
        )
        robot_elapsed_s = [
            self.robot_finish_sim_times_s[robot_id] - self.start_sim_time_s
            for robot_id in self.required_robot_ids
        ] if all_robots_finished else []

        return {
            "mission_id": self.mission_id,
            "controller": self.controller,
            "status": status,
            "success": success,
            "makespan_sim_s": max(task_elapsed_s) if task_elapsed_s else None,
            "mean_task_completion_sim_s": mean(task_elapsed_s) if task_elapsed_s else None,
            "mean_robot_finish_sim_s": mean(robot_elapsed_s) if robot_elapsed_s else None,
            "termination_elapsed_sim_s": end_sim_time_s - self.start_sim_time_s,
            "completed_tasks": len(self.task_completion_sim_times_s),
            "required_tasks": len(self.required_task_ids),
            "completed_robots": len(self.robot_finish_sim_times_s),
            "required_robots": len(self.required_robot_ids),
            "mean_inference_ms": mean(self.inference_times_ms) if self.inference_times_ms else None,
            "p50_inference_ms": _percentile(self.inference_times_ms, 50),
            "p95_inference_ms": _percentile(self.inference_times_ms, 95),
            "p99_inference_ms": _percentile(self.inference_times_ms, 99),
        }
