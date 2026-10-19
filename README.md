# MAPPO Agent Project

This project is a small but complete implementation of Multi-Agent Proximal Policy Optimization (MAPPO) for a cooperative grid-world environment.

The goal is to train several agents to move toward their own goals while sharing the same team reward. The project is intentionally simple so the core ideas of MAPPO are easy to understand:

- each agent has its own local observation
- the critic sees the full global state during training
- agents act in a decentralized way during execution
- PPO is used to update the policy safely and stably

---

## Project overview

This repository is a compact MAPPO implementation for a cooperative multi-agent grid environment.

It is designed to teach the core ideas behind MAPPO in a simple, readable way:

- agents act using local observations
- the critic sees the full environment state during training
- all agents share a common team reward
- PPO updates the policy in a stable and safe way

```text
Environment
   |
   v
Actors (local obs) ----> actions
   |
   +----> shared team reward
   |
   v
Rollout Buffer
   |
   v
GAE + PPO Update
   |
   v
Centralized Critic (global state)
```

## Project idea

This project models a cooperative multi-agent environment where:

- there are N agents
- each agent has its own target goal position
- agents move on a grid with 5 possible actions:
  - stay
  - up
  - down
  - left
  - right
- all agents share the same team reward
- the environment is designed to teach the MAPPO idea of Centralized Training and Decentralized Execution (CTDE)

In short:

- During execution: each agent sees only its own local observation
- During training: the critic sees the full environment state

This is the main idea behind MAPPO.

---

## Why MAPPO is used here

MAPPO is a multi-agent version of PPO. It works by:

1. collecting rollout data from the environment
2. storing observations, actions, rewards, and values
3. computing advantages using GAE (Generalized Advantage Estimation)
4. updating the policy with PPO's clipped objective
5. using a centralized critic to estimate the value of the full state

The actor learns a policy from local observations, while the centralized critic helps estimate whether the current global state is good or bad.

---

## File-by-file explanation

### 1) env.py

This file defines the environment class `SimpleMAEnv`.

#### Responsibilities

- creates the cooperative grid-world
- initializes agent and goal positions
- generates local observations
- generates the global state
- applies actions and returns rewards
- decides when an episode ends

#### Important details

`reset()`
- randomly spawns all agents and goals on the grid
- returns the local observations and the global state

`_get_obs()`
- creates one observation per agent
- each observation contains:
  - own position
  - vector to its own goal
  - relative positions of the other agents

`_get_state()`
- concatenates all agent positions and all goal positions
- this is what the centralized critic sees

`step(actions)`
- moves each agent based on its chosen action
- computes the mean distance between each agent and its goal
- gives a shared team reward
- adds a bonus if all agents reach their goals
- ends the episode when max steps are reached

#### Reward design

The reward is intentionally shaped so that agents are rewarded for getting closer to their goals. A positive bonus is added when all agents finish near their goals. This helps PPO learn much better than a purely negative reward.

---

### 2) networks.py

This file defines the neural networks used by MAPPO.

#### `Actor`

This is the policy network.

- input: local observation for one agent
- output: logits for the action distribution
- uses a small MLP with Tanh activations
- selects actions during training using sampling
- selects greedy action during evaluation

Key methods:

- `forward(obs)`
  - builds a `Categorical` distribution from logits
- `act(obs)`
  - samples an action and returns its log probability
- `act_greedy(obs)`
  - returns the action with the highest probability
- `evaluate(obs, action)`
  - evaluates the chosen action under the current policy
  - returns log probabilities and entropy

#### `CentralizedCritic`

This is the value network.

- input: global state
- output: estimated value of the current state
- uses a small MLP with Tanh activations
- helps PPO estimate how good the current state is

The critic is centralized during training, which means it sees all agent positions and goals, even though the policy only sees local observations.

---

### 3) buffer.py

This file stores rollout data and computes advantages.

#### `RolloutBuffer`

This is the replay memory of the training loop.

It stores:

- `obs`: local observations for each agent
- `state`: global state for each timestep
- `actions`: chosen actions
- `logp`: old policy log probabilities
- `rewards`: team reward for each agent
- `dones`: termination flags
- `values`: critic-predicted values

#### Why this file matters

In PPO, we need old trajectory data before updating. The rollout buffer holds the collected experience from one rollout and then later gives that data to the policy update step.

#### `add(...)`

Adds one timestep to the buffer.

#### `compute_gae(last_values)`

Computes Generalized Advantage Estimation (GAE), which is the standard PPO method for estimating advantages.

The idea is:

- rewards from future steps are discounted
- the critic helps estimate value of future states
- the algorithm balances immediate and future reward information

This gives the policy a better update signal than plain Monte Carlo returns.

#### `get_tensors(...)`

Takes the stored NumPy arrays and converts them to PyTorch tensors so they can be used in training.

It also flattens time and agent dimensions into a single batch, which makes PPO minibatch training easier.

---

### 4) mappo.py

This is the main MAPPO algorithm implementation.

#### `MAPPO.__init__()`

Creates:

- actor network
- centralized critic network
- Adam optimizer

It also sets PPO hyperparameters such as:

- learning rate
- clipping epsilon
- value loss coefficient
- entropy coefficient
- epochs
- minibatch size
- gradient clipping norm

#### `act(obs_list, state)`

Used during rollout collection.

- converts local observations to a tensor
- converts global state to a tensor
- actor selects actions for each agent
- critic evaluates the full state
- returns actions, log probabilities, and values

#### `act_greedy(obs_list)`

Used during evaluation.

This does not sample random actions; it picks the most probable action for each agent.

#### `update(buffer, last_state)`

This is the main PPO update step.

It does the following:

1. bootstrap the final state value with the critic
2. compute advantages and returns with GAE
3. normalize advantages
4. build training tensors from the rollout buffer
5. iterate over PPO epochs and minibatches
6. compute policy loss, critic loss, and entropy bonus
7. combine them into one total loss
8. backpropagate and optimize

#### PPO objective

The actor loss uses the clipped PPO objective:

- compute ratio = new_policy_prob / old_policy_prob
- compute unclipped objective
- compute clipped objective
- use `min(unclipped, clipped)`

This prevents too-large updates and makes PPO stable.

The critic loss uses MSE between predicted value and target return.

The entropy term encourages exploration and prevents the policy from collapsing too early.

---

### 5) train.py

This file contains the training loop and evaluation routine.

#### `evaluate(env, agent, n_episodes, n_steps)`

Runs several episodes using greedy actions and averages their total reward.

This is a useful measurement of how good the policy is without random exploration.

#### `train(...)`

This is the main program loop.

It does the following:

1. set seeds for reproducibility
2. create the environment
3. create the rollout buffer
4. create the MAPPO agent
5. reset the environment
6. for each update:
   - clear the buffer
   - collect a rollout of experience
   - store observations, actions, rewards, dones, values
   - update the agent using MAPPO
   - print learning statistics every 20 updates
7. evaluate the agent without exploration
8. report training metrics

#### Printed metrics

The output line looks like this:

```
update  200 | actor_loss -0.040 | critic_loss 27.184 | train_avg_return 19.974 | eval_avg_return 19.134
```

Meaning:

- `actor_loss`: PPO policy loss
- `critic_loss`: value-network loss
- `train_avg_return`: average return during training episodes
- `eval_avg_return`: average return during greedy evaluation episodes

A good training run should show returns increasing over time, especially the evaluation return.

---

### 6) test_env_reward.py

This is a small regression test file.

It checks that the environment reward is meaningful and positive when an agent is close to its target.

This is important because a bad reward function can make PPO fail even if the code is otherwise correct.

A good reward function should:

- give larger reward when the agent is closer to the goal
- give a strong bonus when the task is completed
- not make the agent receive only negative values all the time

---

## How the whole system works together

The flow is:

1. `train.py` creates a `SimpleMAEnv`
2. the environment gives observations and state to the agent
3. `MAPPO.act()` chooses actions using the actor
4. `env.step(actions)` updates the world and returns rewards and dones
5. the rollout buffer stores the experience
6. after enough steps, `MAPPO.update()` computes GAE and performs PPO updates
7. training continues until a fixed number of update steps is reached
8. evaluation checks policy quality without exploration noise

This is exactly the MAPPO training loop in a compact form.

---

## How to run the project

On Windows with the project venv:

```powershell
.\.venv\Scripts\python.exe train.py
```

Or, if the virtual environment is activated:

```powershell
python train.py
```

---

## How to know if learning is working

Look at these values:

- `train_avg_return`
- `eval_avg_return`

Good signs:

- both numbers rise over time
- evaluation return stays positive and improves
- no huge exploding loss values

Bad signs:

- returns stay near random baseline
- values remain negative for a long time
- evaluation does not improve
- critic loss explodes

---

## Typical interpretation of a successful run

A successful MAPPO run will show something like:

- early training: low returns
- middle training: moderate positive returns
- late training: much better rewards and stable evaluation

This project is a toy example, so you should not expect perfect convergence, but you should expect the agent to improve meaningfully over time.

---

## Summary

This repository is a compact MAPPO implementation that demonstrates the key ideas behind multi-agent reinforcement learning:

- decentralised execution
- centralised training
- PPO policy optimization
- shared team rewards
- rollout buffering and advantage estimation

If you understand the files in this order:

1. `env.py`
2. `networks.py`
3. `buffer.py`
4. `mappo.py`
5. `train.py`

then you understand the whole MAPPO workflow of this project.

---

## Recommended next step

If you want to extend this project later, good improvements would be:

- add more agents
- increase grid size
- add obstacles or walls
- use more realistic reward shaping
- save checkpoints of the trained model
- create a visualization of agent movement

This project is a very good starting point for understanding MAPPO in a small cooperative setting.

## Run the three time and performance metrics

The `time_complexity/` tools report three complementary measurements. They are
mission/performance metrics, **not** a theoretical Big-O analysis:

1. **Mission makespan (simulator seconds):** time from mission start until the last
   required task completes. For mission start `t0` and task completion times `Ci`,
   makespan is `max(Ci) - t0`. It describes total mission duration.
2. **Average completion time (simulator seconds):** mean time from mission start to
   each task (or robot) completion: `sum(Ci - t0) / number_of_completions`. It shows
   how quickly work is completed on average, not just when the final task finishes.
3. **MAPPO inference latency (wall-clock milliseconds):** time for the trained actor
   to produce one batched action decision for the robots. The script reports mean,
   p50, p95, and p99 latency after warm-up. It indicates policy computation cost; it
   does not include training or simulator stepping.

The first two values use simulator timestamps; inference latency uses a wall clock.
Do not compare the numbers directly or treat them as one unit. Use makespan and
average completion time to assess mission speed, and inference latency to assess
whether decisions can be produced within the robot control period. Together they
describe task performance and decision cost. The current repository implements
MAPPO only, so it does **not** yet provide an IPPO/rule-based comparison.

### Files

| File | Purpose |
|---|---|
| `time_complexity/makespan.py` | Calculate makespan from a start timestamp and task completion timestamps. |
| `time_complexity/Average_completion_time.py` | Calculate mean completion offset from mission start. |
| `time_complexity/MAPPO_inference_time.py` | Benchmark warmed actor inference and print mean/p50/p95/p99 latency. |
| `time_complexity/__init__.py` | Marks the metric tools as an importable Python package. |
| `tests/test_time_complexity.py` | Tests timestamp calculations and invalid inputs. |
| `train.py` | Trains MAPPO and saves the actor checkpoint needed by the inference benchmark. |

### Install and run in the VS Code terminal (Windows PowerShell)

Open **Terminal -> New Terminal** in VS Code. These commands create a virtual
environment for your locally selected Python; `.venv` is ignored by Git and must
not be committed. You can use a different supported Python version by selecting
that version when creating the environment. For example, use
`py -3.11 -m venv .venv` (replace `3.11` with your installed version) with the
Windows Python Launcher, or invoke that version's `python.exe` directly.

```powershell
git clone https://github.com/saraabdelfatta/MAPPO-agent-.git
cd MAPPO-agent-
python --version
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If the virtual environment already exists for the Python version you want, skip
`python -m venv .venv`. If it was created with a different Python version, remove
that local `.venv` directory yourself and recreate it with the desired interpreter;
the environment folder is machine-specific and is not part of the repository.

Run the metric tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_time_complexity.py" -v
```

Try the two timestamp calculators with sample data (mission starts at 10 seconds;
the tasks finish at 22, 25, and 28 seconds):

```powershell
.\.venv\Scripts\python.exe -m time_complexity.makespan --start 10 --completion-times 22 25 28
.\.venv\Scripts\python.exe -m time_complexity.Average_completion_time --start 10 --completion-times 22 25 28
```

Expected sample results are makespan **18.000 simulator seconds** and average
completion time **15.000 simulator seconds**. These are example calculations, not
measurements from an experiment. Replace the timestamps with task completion events
from the same scenario when analyzing real runs. Both calculators require absolute
completion timestamps and a mission start timestamp.

Train a short smoke-test policy and save the checkpoint:

```powershell
.\.venv\Scripts\python.exe train.py --updates 2 --steps 8 --checkpoint checkpoints\smoke_test_actor.pt
```

Then benchmark its actor inference:

```powershell
.\.venv\Scripts\python.exe -m time_complexity.MAPPO_inference_time --checkpoint checkpoints\smoke_test_actor.pt --iterations 100 --warmup 10 --device auto
```

For a longer policy run, train the default model and then benchmark its checkpoint:

```powershell
.\.venv\Scripts\python.exe train.py --checkpoint checkpoints\mappo_actor.pt
.\.venv\Scripts\python.exe -m time_complexity.MAPPO_inference_time --checkpoint checkpoints\mappo_actor.pt --iterations 1000 --warmup 100 --device auto
```

The benchmark reports mean, p50, p95, and p99 milliseconds for a batched
three-agent actor decision. `--device auto` selects CUDA when available, otherwise
CPU; use the same device and hardware when comparing runs. A smoke-test checkpoint
only verifies that the benchmark works and is not a research-quality trained
policy. The benchmark currently measures the grid-world actor, not an Isaac Sim
deployment.

The inference benchmark uses a fixed three-agent batch. It reports latency for
that workload; it does not estimate Big-O complexity or measure how inference
scales as robot count and observation size increase. To study scaling, add runs
across multiple agent counts and report latency against robot count on the same
hardware.

### Connecting this prototype to Isaac Sim

This repository currently runs its own `SimpleMAEnv` grid world; it is **not
connected to Isaac Sim**. When integrating Isaac Sim, keep the MAPPO actor and
timing tools separate from simulator-specific code and implement an environment
adapter that:

- launches/attaches to Isaac Sim using an Isaac-supported Python environment,
  opens the selected USD scene, and identifies robot, sensor, and goal prims;
- maps simulator observations to the actor's expected local observation shape and
  maps actor actions to the robots' control commands;
- implements deterministic scenario reset, physics stepping at a documented
  `physics_dt`, and the same action/observation contract as the grid environment;
- reports authoritative task completion, collision, robot failure, and success
  events, and obtains mission start/completion times from the **simulator clock**;
- times actor inference separately with a monotonic wall clock (and synchronizes
  GPU work around the timed call when applicable), excluding simulator stepping;
- records scene/version, robot count, control frequency, seeds, and hardware so
  comparisons can be reproduced.

Do not import Isaac Sim at module import time: load simulator dependencies only
when constructing the Isaac environment so the grid-world tools remain usable
without Isaac Sim installed. First test reset, one action/step, time progression,
completion events, and collision reporting in a single scenario. Then compare
controllers on the same scene, mission, robot configuration, and seeds. The current
inference script constructs `SimpleMAEnv` itself, so Isaac deployment benchmarking
will require adapting its observation source to the simulator adapter while
preserving the same timed actor-only interval.
