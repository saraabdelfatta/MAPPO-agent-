"""
buffer.py
---------
Stores one rollout and computes GAE.

The buffer stores:

    obs       -> local observations for the actor
    state     -> global state for the centralized critic
    actions   -> actions selected by the actors
    logp      -> old policy log probabilities
    rewards   -> team rewards
    dones     -> episode termination flags
    values    -> critic predictions

MAPPO uses the same PPO/GAE machinery as PPO.

The important MAPPO difference is that the critic receives
the GLOBAL state while the actor receives only LOCAL observations.

The environment generates experience → buffer.py stores 
the experience → GAE calculates advantages → MAPPO uses that information to update the policy.
"""

# this file is like the memory of your MAPPO training loop.
#The buffer temporarily stores what happened during a rollout.
import numpy as np
import torch


# Store one batch/rollout of MAPPO experience.
class RolloutBuffer:
    
    #The constructor
    def __init__(
        self,
        n_agents,            #Number of agents
        obs_dim,             #Number of values in one agent's local observation.
        state_dim,           #numbers per global state
        n_steps,             #number of timesteps in one rollout
        gamma=0.99,          #discount factor this show how much the agent cares about future rewards
        lam=0.95             #lambda for GAE this show much the agent cares about future rewards
    ):

        self.n_agents = n_agents
        self.n_steps = n_steps

        self.gamma = gamma
        self.lam = lam

        # ---------------------------------------------------------
        # Storage
        # ---------------------------------------------------------

        # Local observations.
        # (agent sees only its own observation)
        # Shape:
        # (time, agent, observation)
        #
        self.obs = np.zeros(
            (n_steps, n_agents, obs_dim),
            dtype=np.float32
        )

        # Global state.
        # (the possiztion of all agents and goals)
        # Shape:
        # (time, state_dimension)
        #
        self.state = np.zeros(
            (n_steps, state_dim),
            dtype=np.float32
        )

        # Actions.
        #(the agent took this action at this time step)
        # Shape:
        # (time, agent)
        #
        self.actions = np.zeros(
            (n_steps, n_agents),
            dtype=np.int64
        )

        # Log probability of each selected action.
        self.logp = np.zeros(
            (n_steps, n_agents),
            dtype=np.float32
        )

        # Rewards.
        self.rewards = np.zeros(
            (n_steps, n_agents),
            dtype=np.float32
        )

        # Done flags. it is used to controle the agent if the episode is done or not
        self.dones = np.zeros(
            (n_steps, n_agents),
            dtype=np.float32
        )

        # Critic values. it is the value of the state that the critic predicts
        self.values = np.zeros(
            (n_steps, n_agents),
            dtype=np.float32
        )

        # Current position in the buffer.
        self.ptr = 0

    def add(
        self,
        obs,
        state,
        actions,
        logp,
        rewards,
        dones,
        values
    ):
        """
        Store one timestep of experience.
        """

        i = self.ptr

        self.obs[i] = obs
        self.state[i] = state
        self.actions[i] = actions
        self.logp[i] = logp
        self.rewards[i] = rewards
        self.dones[i] = dones
        self.values[i] = values

        self.ptr += 1

    def compute_gae(self, last_values):
        """
        Calculate Generalized Advantage Estimation.

        delta_t =
            r_t
            + gamma * V(s_{t+1}) * (1-done)
            - V(s_t)

        advantage_t =
            delta_t
            + gamma * lambda * (1-done) * advantage_{t+1}
        """

        adv = np.zeros(
            (self.n_steps, self.n_agents),
            dtype=np.float32
        )

        last_adv = np.zeros(
            self.n_agents,
            dtype=np.float32
        )

        # Walk backwards through time.
        for t in reversed(range(self.n_steps)):

            if t == self.n_steps - 1:
                next_value = last_values
            else:
                next_value = self.values[t + 1]

            next_nonterminal = 1.0 - self.dones[t]

            # TD error.
            delta = (
                self.rewards[t]
                + self.gamma * next_value * next_nonterminal
                - self.values[t]
            )

            # GAE recursion.
            last_adv = (
                delta
                + self.gamma
                * self.lam
                * next_nonterminal
                * last_adv
            )

            adv[t] = last_adv

        # PPO value targets.
        returns = adv + self.values

        return adv, returns

    def get_tensors(
        self,
        advantages,
        returns,
        device
    ):
        """
        Convert rollout data into PyTorch tensors.

        We flatten:

            (time, agent)

        into:

            (time * agent)

        This allows PPO to treat each agent timestep as
        one training sample.
        """

        # ---------------------------------------------------------
        # Repeat the global state once for every agent.
        # ---------------------------------------------------------

        state_rep = np.repeat(
            self.state[:, None, :],
            self.n_agents,
            axis=1
        )

        # ---------------------------------------------------------
        # Flatten helper.
        # ---------------------------------------------------------

        def flat(x):

            return torch.as_tensor(
                x.reshape(
                    self.n_steps * self.n_agents,
                    *x.shape[2:]
                ),
                dtype=torch.float32,
                device=device
            )

        # ---------------------------------------------------------
        # Actions need to be LONG integers.
        # ---------------------------------------------------------

        actions = torch.as_tensor(
            self.actions.reshape(
                self.n_steps * self.n_agents
            ),
            dtype=torch.long,
            device=device
        )

        return {
            "obs": flat(self.obs),

            "state": flat(state_rep),

            "actions": actions,

            "old_logp": flat(self.logp),

            "advantages": flat(advantages),

            "returns": flat(returns)
        }

    def reset(self):
        """
        Start writing from the beginning of the buffer.
        """

        self.ptr = 0