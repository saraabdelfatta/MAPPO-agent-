# MAPPO Agent

A compact Python implementation of Multi-Agent Proximal Policy Optimization
(MAPPO) for a cooperative grid-world. It demonstrates **Centralized Training
with Decentralized Execution (CTDE)**: each actor selects actions from its
local observation, while a centralized critic uses the global state during
training.

> **Current scope:** the runnable environment is a small grid-world, not a
> robot simulator. There is no connection to ISSC/Isaac Sim in this repository
> yet. The timing utilities can calculate metrics from data, but they do not
> read events from a simulator.

## Architecture

```text
train.py
  ├── env.py: SimpleMAEnv (local observations, global state, reward, episodes)
  ├── mappo.py
  │     ├── networks.py: decentralized Actor + centralized CentralizedCritic
  │     └── buffer.py: rollout storage + GAE
  └── checkpoint: actor weights saved under checkpoints\

evaluation/timing.py: standalone mission/task/robot/inference metric collector
time_complexity\: command-line metric calculators and actor latency benchmark
tests\: timing and metric tests
```

At each training step, `train.py` asks the MAPPO actor for one action per
agent, advances the environment, and stores the experience in `RolloutBuffer`.
`MAPPO.update()` then calculates advantages with GAE and updates the actor and
critic using PPO. During evaluation, the actor selects greedy actions.

The current grid-world action and data contract is:

| Item | Current definition |
|---|---|
| Agents | 3 by default; configurable with `--agents` |
| Actor input | Own grid position, vector to assigned goal, relative positions of other agents; `4 + 2 × (agents - 1)` values (8 by default) |
| Critic input during training | All agent positions and goal positions; `4 × agents` values (12 by default) |
| Actions | `0` stay, `1` up, `2` down, `3` left, `4` right |
| Reward | Shared positive distance-based reward, with a bonus when all agents reach their goals |
| Episode end | Fixed maximum of 25 environment steps |

The grid-world has no obstacles, physics, sensor model, task queue, collision
handling, or simulator-clock. Its reward is not mission completion time.

## Project files

| Path | Purpose |
|---|---|
| `env.py` | Implements `SimpleMAEnv`: creates agent/goal positions, returns local observations and global state, applies discrete moves, calculates shared rewards, and signals the step limit. |
| `networks.py` | Defines `Actor` (local observation to action distribution) and `CentralizedCritic` (global state to value estimate). |
| `buffer.py` | Defines `RolloutBuffer`, which stores rollout observations, states, actions, rewards, log probabilities, done flags, and values; calculates GAE and converts samples to PyTorch tensors. |
| `mappo.py` | Implements action selection and PPO/MAPPO updates, including clipped policy loss, critic value loss, entropy regularization, and gradient clipping. |
| `train.py` | Training entry point; collects rollouts, updates MAPPO, prints periodic evaluation metrics, and saves the trained actor checkpoint. |
| `requirements.txt` | Lists the third-party Python packages required by the project. |
| `checkpoints\mappo_actor.pt` | Example saved actor checkpoint present in this workspace. A training run can create or overwrite the default checkpoint. |
| `evaluation\timing.py` | Simulator-independent `MissionTimingEvaluator`; accepts explicit task/robot completion times and actor inference durations, then reports mission metrics. No simulator events are collected automatically. |
| `time_complexity\makespan.py` | Calculates mission makespan from a mission start time and task completion timestamps. |
| `time_complexity\Average_completion_time.py` | Calculates average completion time from a mission start time and completion timestamps. |
| `time_complexity\MAPPO_inference_time.py` | Measures batched greedy actor latency for the current three-agent grid-world using a trained actor checkpoint. This is not an Isaac Sim benchmark. |
| `tests\test_timing.py` | Tests success/timeout timing results, event validation, and inference metrics. |
| `tests\test_time_complexity.py` | Tests the standalone time metric calculators. |
| `test_env_reward.py` | Root-level regression test for the grid-world reward. |

The current training script saves the actor weights and its input/action
dimensions; it does not save the centralized critic or optimizer state for
resuming training.

## ISSC / Isaac Sim integration status

If “ISSC Sim” refers to **NVIDIA Isaac Sim**, it is **not currently connected**.
There is no Isaac Sim adapter, ROS 2 bridge, simulator configuration, robot
controller, or Isaac Sim dependency in the project. The only implemented
environment is `SimpleMAEnv` in `env.py`.

The existing actor and grid-world cannot be connected to a robot merely by
loading the checkpoint: Isaac Sim observations would need to match the actor's
expected observation dimensions and meanings (8 values for the default
three-agent setup), and its continuous robot controls would need an explicit
mapping from the policy's five discrete actions.

A future integration should add an Isaac Sim environment adapter that:

1. Resets the selected scene and robots, and obtains the robot poses, task
   state, and any sensor data required by the agreed observation design.
2. Converts each robot's simulator data into the local-observation format
   expected by the actor. During training, it also provides a global state to
   the centralized critic. Do not expose global information to deployed actors
   unless those robots can actually observe it.
3. Maps each selected action to a documented robot-control command and
   advances the simulator at a defined control frequency.
4. Returns observations, rewards, termination status, and authoritative task
   or robot completion events to a training/evaluation runner.
5. Sends simulator-clock completion events and measured actor inference
   durations to `MissionTimingEvaluator` for consistent reporting.

Before using the existing policy in a simulator, define and test the observation
and action contracts, robot control mapping, task-success predicate, collision
rules, and reward. If the observation/action design changes, adapt and retrain
the policy as needed. Isaac Sim and any ROS 2 packages should only be installed
when that adapter is implemented; they are not needed to run the code currently
in this repository.

## Dependencies

- **Python 3.10 or newer**
- **NumPy 1.24 or newer**
- **PyTorch 2.0 or newer**

The Python standard library is used for the other functionality. Dependency
versions are declared in `requirements.txt`. GPU support is optional; training
and inference can run on CPU. If using CUDA, install a PyTorch build compatible
with the installed NVIDIA driver and CUDA runtime.

## Install and run (Windows PowerShell)

Open a PowerShell terminal in the project folder, or change to it:

```powershell
cd "E:\AUC\theis 2\new semaster\MAPPO Agent"
```

Create a virtual environment and install dependencies (once per environment):

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Use Python 3.10 or newer. If the `python` command is not found, install Python
and reopen PowerShell. Activating `.venv` is optional; the commands below invoke
its Python executable directly.

### Run tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
.\.venv\Scripts\python.exe -m unittest test_env_reward -v
```

### Train MAPPO

Run the default training job (800 updates, 64 environment steps per update,
3 agents, seed 0):

```powershell
.\.venv\Scripts\python.exe train.py
```

The actor checkpoint is saved to `checkpoints\mappo_actor.pt`. A short smoke
run can check that training and checkpoint saving work:

```powershell
.\.venv\Scripts\python.exe train.py --updates 2 --steps 8 --checkpoint checkpoints\smoke_test_actor.pt
```

Available options are `--updates`, `--steps`, `--agents`, `--seed`, and
`--checkpoint`; run `.\.venv\Scripts\python.exe train.py --help` for details.
Training reports actor/critic losses and average training/evaluation returns
every 20 updates.

### Benchmark actor inference

After training, run the actor-only latency benchmark:

```powershell
.\.venv\Scripts\python.exe -m time_complexity.MAPPO_inference_time --checkpoint checkpoints\mappo_actor.pt --iterations 1000 --warmup 100
```

The benchmark reports mean, p50, p95, and p99 latency for a batched decision
covering the current three agents. It accepts `--device auto`, `--device cpu`,
or `--device cuda`. Use the same hardware and device when comparing policies.
This measures grid-world actor inference only: it does not include simulator
stepping, critic training, or an Isaac Sim integration.

### Calculate time metrics from supplied timestamps

These utilities calculate metrics from timestamps passed on the command line;
they do not collect timestamps from a simulator.

```powershell
.\.venv\Scripts\python.exe -m time_complexity.makespan --start 0 --completion-times 12 15 18
.\.venv\Scripts\python.exe -m time_complexity.Average_completion_time --start 0 --completion-times 12 15 18
```

The examples produce a makespan of 18 seconds and an average completion time
of 15 seconds. They are sample inputs, not measured experimental results.
Provide absolute simulator timestamps when using actual experiment data; `--start`
is the mission start timestamp.

## Timing evaluator

`evaluation\timing.py` provides a reusable `MissionTimingEvaluator`, but no
runner currently connects it to an environment or simulator. A future runner
should create one evaluator per mission, record simulator-confirmed task and
robot completion timestamps, record the wall-clock duration of each actor
decision, and call `result()` at episode end. The evaluator distinguishes
simulator time from wall-clock inference time and does not report successful
mission makespan or average task completion for failed/timeout episodes.

For meaningful comparisons, use identical scenarios, tasks, robot counts,
initial poses, simulator settings, and evaluation seeds. Report success rate
and failures alongside makespan, completion time, collisions, and inference
latency. Do not treat example timestamps or grid-world rewards as simulator
results.
