"""
mappo.py
--------
Implementation of MAPPO.

MAPPO combines:

    PPO
    +
    Centralized Training
    +
    Decentralized Execution

The actor sees LOCAL observations.

The centralized critic sees the GLOBAL state.
mappo.py is responsible for making the agents act, evaluating their actions, and updating 
the Actor and Critic using PPO.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from networks import Actor, CentralizedCritic


class MAPPO:

    def __init__(
        self,
        obs_dim,
        state_dim,
        n_actions,
        device="cpu",
        lr=5e-4,
        clip_eps=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
        epochs=8,
        minibatch_size=64,
        max_grad_norm=0.5
    ):

        self.device = device

        # PPO clipping parameter.
        self.clip_eps = clip_eps

        # Weight of critic loss.
        self.vf_coef = vf_coef

        # Weight of entropy bonus.
        self.ent_coef = ent_coef

        # Number of PPO passes over the rollout.
        self.epochs = epochs

        # Minibatch size.
        self.minibatch_size = minibatch_size

        # Gradient clipping.
        self.max_grad_norm = max_grad_norm

        # ---------------------------------------------------------
        # Actor
        # ---------------------------------------------------------

        self.actor = Actor(
            obs_dim,
            n_actions
        ).to(device)

        # ---------------------------------------------------------
        # Centralized critic
        # ---------------------------------------------------------

        self.critic = CentralizedCritic(
            state_dim
        ).to(device)

        # ---------------------------------------------------------
        # Optimizer
        # ---------------------------------------------------------

        self.optimizer = optim.Adam(
            list(self.actor.parameters())
            +
            list(self.critic.parameters()),
            lr=lr
        )

    @torch.no_grad()
    def act(
        self,
        obs_list,
        state
    ):
        """
        Select actions during rollout collection.

        Actor:
            local observations

        Critic:
            global state
        """

        # Convert all agent observations into one tensor.
        #
        # Shape:
        #       (n_agents, obs_dim)
        obs_t = torch.as_tensor(
            np.array(obs_list),
            dtype=torch.float32,
            device=self.device
        )

        # Convert global state into a batch of size 1.
        #
        # Shape:
        #       (1, state_dim)
        state_t = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

        # ---------------------------------------------------------
        # Actor
        # ---------------------------------------------------------

        # Each agent selects an action using its own local obs.
        actions, logp = self.actor.act(obs_t)

        # ---------------------------------------------------------
        # Critic
        # ---------------------------------------------------------

        # One value for the entire global state.
        value = self.critic(state_t)

        # value shape:
        #       (1,)
        #
        # Repeat it once for each agent so that it matches
        # the buffer format.
        value = value.expand(
            obs_t.shape[0]
        )

        return (
            actions.cpu().numpy(),
            logp.cpu().numpy(),
            value.cpu().numpy()
        )

    @torch.no_grad()
    def act_greedy(self, obs_list):
        """
        Evaluation mode.

        No random exploration.
        Each agent chooses its highest-logit action.
        """

        obs_t = torch.as_tensor(
            np.array(obs_list),
            dtype=torch.float32,
            device=self.device
        )

        actions = self.actor.act_greedy(obs_t)

        return actions.cpu().numpy()

    def update(
        self,
        buffer,
        last_state
    ):
        """
        Perform the PPO/MAPPO update.
        """

        # =========================================================
        # STEP 1
        # Bootstrap the value after the final rollout state.
        # =========================================================

        with torch.no_grad():

            last_state_t = torch.as_tensor(
                last_state,
                dtype=torch.float32,
                device=self.device
            ).unsqueeze(0)

            last_value = self.critic(
                last_state_t
            ).item()

            # Same centralized value is used for every agent.
            last_values = np.full(
                buffer.n_agents,
                last_value,
                dtype=np.float32
            )

        # =========================================================
        # STEP 2
        # Calculate GAE.
        # =========================================================

        advantages, returns = buffer.compute_gae(
            last_values
        )

        # =========================================================
        # STEP 3
        # Normalize advantages.
        # =========================================================

        advantages = (
            advantages - advantages.mean()
        ) / (
            advantages.std() + 1e-8
        )

        # =========================================================
        # STEP 4
        # Convert everything to tensors.
        # =========================================================

        data = buffer.get_tensors(
            advantages,
            returns,
            self.device
        )

        n_samples = data["obs"].shape[0]

        # Variables for final statistics.
        actor_loss_value = 0.0
        critic_loss_value = 0.0

        # =========================================================
        # STEP 5
        # PPO update.
        # =========================================================

        for _ in range(self.epochs):

            # Shuffle the training samples.
            indices = torch.randperm(
                n_samples,
                device=self.device
            )

            # Create minibatches.
            for start in range(
                0,
                n_samples,
                self.minibatch_size
            ):

                mb_indices = indices[
                    start:start + self.minibatch_size
                ]

                # -------------------------------------------------
                # Get minibatch.
                # -------------------------------------------------

                mb_obs = data["obs"][mb_indices]

                mb_state = data["state"][mb_indices]

                mb_actions = data["actions"][mb_indices]

                mb_old_logp = data["old_logp"][mb_indices]

                mb_adv = data["advantages"][mb_indices]

                mb_returns = data["returns"][mb_indices]

                # =================================================
                # ACTOR / POLICY LOSS
                # =================================================

                new_logp, entropy = self.actor.evaluate(
                    mb_obs,
                    mb_actions
                )

                # PPO importance-sampling ratio:
                #
                #       new policy probability
                #       -----------------------
                #       old policy probability
                #
                ratio = torch.exp(
                    new_logp - mb_old_logp
                )

                # Unclipped PPO objective.
                surr1 = ratio * mb_adv

                # Clipped PPO objective.
                surr2 = (
                    torch.clamp(
                        ratio,
                        1 - self.clip_eps,
                        1 + self.clip_eps
                    )
                    * mb_adv
                )

                # PPO uses the minimum objective.
                actor_loss = -torch.min(
                    surr1,
                    surr2
                ).mean()

                # =================================================
                # CRITIC / VALUE LOSS
                # =================================================

                # IMPORTANT:
                # The critic receives GLOBAL state.
                new_values = self.critic(
                    mb_state
                )

                critic_loss = nn.functional.mse_loss(
                    new_values,
                    mb_returns
                )

                # =================================================
                # ENTROPY
                # =================================================

                entropy_loss = -entropy.mean()

                # =================================================
                # TOTAL LOSS
                # =================================================

                loss = (
                    actor_loss
                    + self.vf_coef * critic_loss
                    + self.ent_coef * entropy_loss
                )

                # =================================================
                # BACKPROPAGATION
                # =================================================

                self.optimizer.zero_grad()

                loss.backward()

                # Prevent very large gradients.
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters())
                    +
                    list(self.critic.parameters()),
                    self.max_grad_norm
                )

                self.optimizer.step()

                # Save latest statistics.
                actor_loss_value = actor_loss.item()
                critic_loss_value = critic_loss.item()

        return {
            "actor_loss": actor_loss_value,
            "critic_loss": critic_loss_value
        }