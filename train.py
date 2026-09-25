"""
train.py
--------
Main MAPPO training program.

Training process:

    1. Reset environment
    2. Collect rollout
    3. Store observations, actions, rewards, values
    4. Compute GAE
    5. Perform PPO/MAPPO update
    6. Repeat

Run this file to train the MAPPO agent.
"""

import numpy as np
import torch

from env import SimpleMAEnv
from buffer import RolloutBuffer
from mappo import MAPPO


def evaluate(
    env,
    agent,
    n_episodes=10,
    n_steps=64
):
    """
    Evaluate the current policy.

    During evaluation we use greedy actions instead of
    stochastic actions.

    This means there is no exploration noise.
    """

    episode_returns = []

    for _ in range(n_episodes):

        obs, state = env.reset()

        total_reward = 0.0

        for _ in range(n_steps):

            # Greedy action selection.
            actions = agent.act_greedy(obs)

            # Environment step.
            obs, state, rewards, dones = env.step(
                actions
            )

            # All agents have the same reward,
            # so rewards[0] represents the team reward.
            total_reward += rewards[0]

            if dones[0]:
                break

        episode_returns.append(
            total_reward
        )

    return float(
        np.mean(episode_returns)
    )


def train(
    n_updates=800,
    n_steps=64,
    n_agents=3,
    seed=0
):
    """
    Main MAPPO training loop.
    """

    # ---------------------------------------------------------
    # Reproducibility
    # ---------------------------------------------------------

    torch.manual_seed(seed)
    np.random.seed(seed)

    # ---------------------------------------------------------
    # Create environment
    # ---------------------------------------------------------

    env = SimpleMAEnv(
        n_agents=n_agents,
        seed=seed
    )

    # ---------------------------------------------------------
    # Create rollout buffer
    # ---------------------------------------------------------

    buffer = RolloutBuffer(
        n_agents=n_agents,
        obs_dim=env.obs_dim,
        state_dim=env.state_dim,
        n_steps=n_steps
    )

    # ---------------------------------------------------------
    # Select device
    # ---------------------------------------------------------

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Using device: {device}"
    )

    # ---------------------------------------------------------
    # Create MAPPO agent
    # ---------------------------------------------------------

    agent = MAPPO(
        obs_dim=env.obs_dim,
        state_dim=env.state_dim,
        n_actions=env.n_actions,
        device=device
    )

    # ---------------------------------------------------------
    # Reset environment
    # ---------------------------------------------------------

    obs, state = env.reset()

    # Current episode reward.
    episode_return = 0.0

    # Store completed episode rewards.
    returns_log = []

    # =========================================================
    # MAIN TRAINING LOOP
    # =========================================================

    for update in range(n_updates):

        # Empty the old rollout.
        buffer.reset()

        # -----------------------------------------------------
        # Collect rollout
        # -----------------------------------------------------

        for _ in range(n_steps):

            # Actor selects actions using LOCAL observations.
            #
            # Critic evaluates GLOBAL state.
            actions, logp, values = agent.act(
                obs,
                state
            )

            # Environment executes actions.
            (
                next_obs,
                next_state,
                rewards,
                dones
            ) = env.step(actions)

            # Store this timestep.
            buffer.add(
                obs=obs,
                state=state,
                actions=actions,
                logp=logp,
                rewards=rewards,
                dones=dones,
                values=values
            )

            # Since this is a shared team reward,
            # rewards[0] is enough.
            episode_return += rewards[0]

            # Move to next timestep.
            obs = next_obs
            state = next_state

            # -------------------------------------------------
            # Episode finished?
            # -------------------------------------------------

            if dones[0]:

                returns_log.append(
                    episode_return
                )

                episode_return = 0.0

                # Start a new episode.
                obs, state = env.reset()

        # -----------------------------------------------------
        # Update MAPPO using collected rollout.
        # -----------------------------------------------------

        stats = agent.update(
            buffer,
            last_state=state
        )

        # -----------------------------------------------------
        # Print progress every 20 updates.
        # -----------------------------------------------------

        if (update + 1) % 20 == 0:

            if returns_log:

                avg_return = np.mean(
                    returns_log[-10:]
                )

            else:

                avg_return = float("nan")

            # Evaluate without exploration.
            eval_return = evaluate(
                env,
                agent,
                n_episodes=10,
                n_steps=n_steps
            )

            print(
                f"update {update + 1:4d} | "
                f"actor_loss {stats['actor_loss']:.3f} | "
                f"critic_loss {stats['critic_loss']:.3f} | "
                f"train_avg_return {avg_return:.3f} | "
                f"eval_avg_return {eval_return:.3f}"
            )

    return agent


# =============================================================
# PROGRAM ENTRY POINT
# =============================================================

if __name__ == "__main__":

    train()