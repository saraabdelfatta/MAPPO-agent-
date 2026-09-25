"""
env.py
------
A tiny COOPERATIVE multi-agent environment written from scratch.

N agents move on a grid and must each end up close to a goal.
Everyone shares the SAME team reward.

This demonstrates the main MAPPO idea:
CTDE = Centralized Training + Decentralized Execution.

During execution:
    Actor -> sees only local observation

During training:
    Critic -> sees the global state
"""

import numpy as np


class SimpleMAEnv:
    def __init__(self, n_agents=3, grid_size=5, max_steps=25, seed=0):

        self.n_agents = n_agents
        self.grid_size = grid_size
        self.max_steps = max_steps

        self.rng = np.random.default_rng(seed)

        # 5 possible actions:
        # 0 = stay
        # 1 = up
        # 2 = down
        # 3 = left
        # 4 = right
        self.n_actions = 5

        self._deltas = np.array([
            [0, 0],
            [0, 1],
            [0, -1],
            [-1, 0],
            [1, 0]
        ])

        # Local observation for each agent:
        #
        # [own_x, own_y,
        #  goal_dx, goal_dy,
        #  other1_dx, other1_dy,
        #  other2_dx, other2_dy, ...]
        #
        # 4 values + 2 values for every other agent
        self.obs_dim = 4 + 2 * (n_agents - 1)

        # Global state:
        #
        # all agent positions + all goal positions
        #
        # agent positions = 2 * n_agents
        # goal positions  = 2 * n_agents
        self.state_dim = 4 * n_agents

    def reset(self):
        """
        Start a new episode.
        """

        # Randomly place agents on the grid.
        self.agent_pos = self.rng.integers(
            0,
            self.grid_size,
            size=(self.n_agents, 2)
        ).astype(np.float32)

        # Randomly place goals on the grid.
        self.goal_pos = self.rng.integers(
            0,
            self.grid_size,
            size=(self.n_agents, 2)
        ).astype(np.float32)

        self.t = 0

        return self._get_obs(), self._get_state()

    def _get_obs(self):
        """
        Create one LOCAL observation for every agent.
        """

        obs_list = []

        for i in range(self.n_agents):

            # Current position of this agent.
            own = self.agent_pos[i]

            # Direction from agent to its own goal.
            goal_vec = self.goal_pos[i] - own

            # Relative positions of all other agents.
            others = []

            for j in range(self.n_agents):

                if j == i:
                    continue

                others.append(self.agent_pos[j] - own)

            if others:
                others = np.concatenate(others)
            else:
                others = np.array([], dtype=np.float32)

            # Combine everything into one local observation.
            observation = np.concatenate([
                own,
                goal_vec,
                others
            ]).astype(np.float32)

            obs_list.append(observation)

        return obs_list

    def _get_state(self):
        """
        Create the GLOBAL state.

        The centralized critic can see:
            - all agent positions
            - all goal positions
        """

        return np.concatenate([
            self.agent_pos.flatten(),
            self.goal_pos.flatten()
        ]).astype(np.float32)

    def step(self, actions):
        """
        Move every agent according to its selected action.
        """

        for i, action in enumerate(actions):

            self.agent_pos[i] = np.clip(
                self.agent_pos[i] + self._deltas[action],
                0,
                self.grid_size - 1
            )

        # Calculate distance between every agent and its own goal.
        distances = np.linalg.norm(
            self.agent_pos - self.goal_pos,
            axis=1
        )

        # Shared team reward.
        #
        # Use a dense, positive reward that increases as agents get
        # closer to their goals. This gives PPO a much clearer learning
        # signal than a purely negative distance penalty.
        max_distance = np.sqrt(2.0) * (self.grid_size - 1)
        reward = 1.0 - (distances.mean() / max_distance)

        # Extra bonus if EVERY agent reaches its goal.
        if np.all(distances < 0.5):
            reward += 5.0

        self.t += 1

        # Episode ends when max_steps is reached.
        done = self.t >= self.max_steps

        # Every agent receives the SAME team reward.
        rewards = [reward] * self.n_agents

        # Every agent receives the same done signal.
        dones = [done] * self.n_agents

        return (
            self._get_obs(),
            self._get_state(),
            rewards,
            dones
        )