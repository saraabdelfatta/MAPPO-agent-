"""
networks.py
-----------
Contains the two neural networks used by MAPPO.

1. Actor
   -------
   Receives LOCAL observation.

   observation -> action probabilities

   This is the decentralized part of MAPPO.


2. CentralizedCritic
   -----------------
   Receives GLOBAL state.

   global state -> value V(s)

   This is the centralized training part of MAPPO.
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical


class Actor(nn.Module):

    def __init__(
        self,
        obs_dim,
        n_actions,
        hidden=64
    ):

        super().__init__()

        # ---------------------------------------------------------
        # Actor neural network.
        #
        # Input:
        #       local observation
        #
        # Output:
        #       one logit for each possible action
        # ---------------------------------------------------------

        self.net = nn.Sequential(

            nn.Linear(obs_dim, hidden),
            nn.Tanh(),

            nn.Linear(hidden, hidden),
            nn.Tanh(),

            nn.Linear(hidden, n_actions)
        )

    def forward(self, obs):
        """
        Convert observations into a categorical probability
        distribution over actions.
        """

        logits = self.net(obs)

        return Categorical(logits=logits)

    def act(self, obs):
        """
        Select actions during training.

        Actions are sampled, which provides exploration.
        """

        dist = self.forward(obs)

        action = dist.sample()

        logp = dist.log_prob(action)

        return action, logp

    def act_greedy(self, obs):
        """
        Select the action with the highest probability.

        Used during evaluation only.
        """

        logits = self.net(obs)

        return torch.argmax(
            logits,
            dim=-1
        )

    def evaluate(self, obs, action):
        """
        Evaluate actions using the CURRENT policy.

        Used during PPO training.
        """

        dist = self.forward(obs)

        logp = dist.log_prob(action)

        entropy = dist.entropy()

        return logp, entropy


class CentralizedCritic(nn.Module):

    def __init__(
        self,
        state_dim,
        hidden=64
    ):

        super().__init__()

        # ---------------------------------------------------------
        # Centralized critic.
        #
        # Input:
        #       GLOBAL state
        #
        # Output:
        #       V(s)
        # ---------------------------------------------------------

        self.net = nn.Sequential(

            nn.Linear(state_dim, hidden),
            nn.Tanh(),

            nn.Linear(hidden, hidden),
            nn.Tanh(),

            nn.Linear(hidden, 1)
        )

    def forward(self, state):

        # Remove the final dimension.
        #
        # Before:
        #       (batch, 1)
        #
        # After:
        #       (batch,)
        return self.net(state).squeeze(-1)