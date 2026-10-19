"""Benchmark batched greedy actor inference using a trained MAPPO checkpoint."""

import argparse
import time
from statistics import mean

import numpy as np
import torch

from env import SimpleMAEnv
from networks import Actor


def percentile(values: list[float], percent: float) -> float:
    """Return one percentile of measured latency values."""
    return float(np.percentile(values, percent))


def main() -> None:
    """Load a trained actor, warm it up, and report inference latency statistics."""
    parser = argparse.ArgumentParser(
        description="Measure MAPPO actor decision latency on the grid-world prototype."
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/mappo_actor.pt",
        help="Actor checkpoint produced by train.py.",
    )
    parser.add_argument("--iterations", type=int, default=1000, help="Timed decisions to measure.")
    parser.add_argument("--warmup", type=int, default=100, help="Untimed warm-up decisions.")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device. Use the same device when comparing algorithms.",
    )
    args = parser.parse_args()
    if args.iterations < 1 or args.warmup < 0:
        parser.error("--iterations must be positive and --warmup cannot be negative")

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available in this PyTorch installation")
    device = (
        "cuda"
        if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    except FileNotFoundError:
        parser.error(
            f"Checkpoint not found: {args.checkpoint}. Train first with `python train.py`."
        )

    env = SimpleMAEnv(n_agents=3, seed=123)
    if checkpoint["obs_dim"] != env.obs_dim or checkpoint["n_actions"] != env.n_actions:
        parser.error("Checkpoint dimensions do not match the current three-agent grid-world environment")
    actor = Actor(env.obs_dim, env.n_actions).to(device)
    actor.load_state_dict(checkpoint["actor_state_dict"])
    actor.eval()

    def decide(observations: list[np.ndarray]) -> None:
        """Run one batched greedy decision for all agents."""
        obs_tensor = torch.as_tensor(
            np.asarray(observations), dtype=torch.float32, device=device
        )
        with torch.inference_mode():
            actions = actor.act_greedy(obs_tensor)
            actions.cpu().numpy()

    for _ in range(args.warmup):
        observations, _ = env.reset()
        decide(observations)

    latency_ms = []
    for _ in range(args.iterations):
        observations, _ = env.reset()
        if device == "cuda":
            torch.cuda.synchronize()
        start_ns = time.perf_counter_ns()
        decide(observations)
        if device == "cuda":
            torch.cuda.synchronize()
        latency_ms.append((time.perf_counter_ns() - start_ns) / 1_000_000)

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}; agents per decision: {env.n_agents}")
    print(f"Timed decisions: {args.iterations}; warm-up decisions: {args.warmup}")
    print(f"Mean inference latency: {mean(latency_ms):.4f} ms")
    print(f"p50 inference latency: {percentile(latency_ms, 50):.4f} ms")
    print(f"p95 inference latency: {percentile(latency_ms, 95):.4f} ms")
    print(f"p99 inference latency: {percentile(latency_ms, 99):.4f} ms")
    print(
        "Prototype benchmark only: this measures the trained grid-world actor, "
        "not an Isaac Sim deployment."
    )


if __name__ == "__main__":
    main()
