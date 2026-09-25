import unittest

import numpy as np

from env import SimpleMAEnv


class RewardTests(unittest.TestCase):
    def test_reward_should_be_positive_when_agent_is_close_to_goal(self):
        env = SimpleMAEnv(n_agents=1, seed=0)
        env.agent_pos = np.array([[1.0, 0.0]], dtype=np.float32)
        env.goal_pos = np.array([[0.0, 0.0]], dtype=np.float32)
        env.t = 0

        _, _, rewards, _ = env.step(np.array([0]))

        self.assertGreater(rewards[0], 0.0)


if __name__ == "__main__":
    unittest.main()
