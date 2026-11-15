# Presentation Script: MAPPO Training Cycle

## Suggested 1-minute explanation

This diagram shows the MAPPO training cycle implemented in our current grid-world prototype. It is separate from the CyberScape integration architecture.

Training starts by creating the environment, rollout buffer, and MAPPO agent. The program then collects a fixed-length rollout. At every environment step, the actor receives each agent's local observation and samples an action. The environment applies all actions and returns a shared team reward and done flags. The rollout buffer stores the observations, global state, actions, old action probabilities, rewards, done flags, and critic values.

The actor is decentralized because it chooses actions from local observations. The centralized critic reads the global state during training and estimates the team value. In this implementation, the same actor network weights are applied to each agent.

After the configured number of steps, MAPPO bootstraps the final state value and computes GAE advantages and return targets. It then uses PPO to update both networks: the actor uses the clipped PPO objective, and the critic learns with a value-prediction loss. Training repeats with the updated policy and a fresh rollout.

Every 20 updates, the program evaluates the policy with greedy actions over 10 episodes. When training finishes, it saves the actor weights as a checkpoint.

This demonstrates the MAPPO learning cycle, but it is still a grid-world implementation. It does not yet train in Isaac Sim or connect to CyberScape, ROS 2, or robot sensors and controls.

## Cycle in one line

**Initialize → collect rollout → compute GAE → update actor and critic → collect the next rollout with the updated policy.**
