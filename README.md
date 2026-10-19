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

## Research architecture and next steps

### What this repository does and does not demonstrate

The current implementation is a useful MAPPO learning prototype, but its `SimpleMAEnv`
is a grid world: it has no obstacles, physical robot dynamics, task queue, collision
model, Isaac Sim connection, or mission-time evaluator. Its episode reward is not a
mission-completion-time measurement. Do not use its current rewards or training output
as evidence that MAPPO improves a real or simulated robot mission.

Keep the existing prototype runnable while developing the research system around
stable interfaces. A suitable target layout is:

```text
configs/
  base.yaml
  gridworld.yaml
  isaac_sim.yaml
envs/
  base.py                 # reset/step/close and observation/action contract
  gridworld.py            # current SimpleMAEnv, adapted to the interface
  isaac_sim/              # Isaac Sim adapter; simulator-specific code stays here
algorithms/
  mappo.py
  ippo.py
  rule_based.py
runner/
  train.py
  evaluate.py
evaluation/
  timing.py               # event-based metrics; no policy or simulator dependency
  compare.py              # matched-scenario aggregation and reporting
  schemas.py              # episode, task, robot, and inference records
utils/
  checkpoint.py
  logging.py
tests/
  test_timing.py
  test_env_contract.py
train.py
evaluate.py
```

Refactor in small steps rather than moving every file at once. First define and test
the environment interface; then adapt the grid world; then add evaluation and baseline
controllers; connect Isaac Sim only when its API and robot/task definitions are
available. Training code should depend on the environment interface, not on Isaac Sim
imports. Isaac Sim is the simulator; it is not another name for the MAPPO algorithm.

### MAPPO contract: inputs, actions, and reward

The current grid-world contract is:

- **Actor observation:** each robot's own position, the relative vector to its assigned
  goal, and relative positions of other robots. There is currently no obstacle or
  task-status information.
- **Training critic input:** the global vector of all robot positions and all goal
  positions. This is available to the centralized critic during training only.
- **Actions:** five discrete moves: stay, up, down, left, right.
- **Reward:** shared team reward based on mean distance to goals, with a team bonus
  when all robots reach their goals. This is a prototype shaping reward, not elapsed
  mission time.

Before training in a robotics simulator, specify and version the actual observation
fields (including sensor/obstacle and task status as needed), action meaning and
control frequency, task-completion predicate, collision rules, and reward terms.
Do not give the decentralized actors global simulator state if the intended deployed
robots will not have it.

### Install, train, and run the timing tools (Windows)

This prototype needs Python 3.10 or newer. The current MAPPO grid-world code requires
NumPy and PyTorch; these dependencies are listed in [`requirements.txt`](./requirements.txt).
Isaac Sim is **not** required for the grid-world steps below and is not connected to
this project yet.

Open PowerShell in the project folder, or change to it first:

```powershell
cd "E:\AUC\theis 2\new semaster\MAPPO Agent"
```

Create a virtual environment and install dependencies (run this once):

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If `python --version` reports an older Python version or Python is not found, install
Python 3.10+ first, then reopen PowerShell. You can run every command below using
`.\.venv\Scripts\python.exe`; activating the virtual environment is optional.

Run the tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
.\.venv\Scripts\python.exe -m unittest test_env_reward -v
```

Train MAPPO and save its actor checkpoint to `checkpoints\mappo_actor.pt`:

```powershell
.\.venv\Scripts\python.exe train.py
```

The default training run uses 800 updates and may take a while. To check that the
pipeline works before starting the full run:

```powershell
.\.venv\Scripts\python.exe train.py --updates 2 --steps 8 --checkpoint checkpoints\smoke_test_actor.pt
```

Then benchmark actor inference from that saved prototype checkpoint:

```powershell
.\.venv\Scripts\python.exe -m time_complexity.MAPPO_inference_time --checkpoint checkpoints\smoke_test_actor.pt --iterations 100 --warmup 10
```

For a research-quality MAPPO inference result, first train the full policy and use
`checkpoints\mappo_actor.pt` instead. The script measures batched actor decision
latency for the current three-agent grid world, reports mean/p50/p95/p99 milliseconds,
and can use `--device cpu` or `--device cuda`. Record the hardware and device; compare
methods on the same hardware. A short smoke-test policy is only for checking the
measurement command, not for reporting algorithm performance.

### Run the separate time-metric calculators

Each file in [`time_complexity/`](./time_complexity/) can be run independently as a
Python module from the project root:

```powershell
.\.venv\Scripts\python.exe -m time_complexity.makespan --start 0 --completion-times 12 15 18
.\.venv\Scripts\python.exe -m time_complexity.Average_completion_time --start 0 --completion-times 12 15 18
.\.venv\Scripts\python.exe -m time_complexity.MAPPO_inference_time --checkpoint checkpoints\mappo_actor.pt --iterations 1000 --warmup 100
```

The first illustrative input prints a makespan of 18 seconds; the second prints an
average completion time of 15 seconds. These supplied timestamps are **example inputs,
not measured research results**. For your experiments, replace them with task or robot
completion timestamps from the same simulator scenario. Pass absolute simulator
timestamps; `--start` is the mission start timestamp. The first two tools calculate
metrics from supplied times; they do not collect events from a simulator.

| Metric script | Calculates | Required input |
|---|---|---|
| `time_complexity/makespan.py` | Last required task completion minus mission start | Mission start and task completion timestamps |
| `time_complexity/Average_completion_time.py` | Mean completion offset from mission start | Mission start and task or robot finish timestamps |
| `time_complexity/MAPPO_inference_time.py` | Mean, p50, p95, and p99 wall-clock actor latency | Trained actor checkpoint and iteration count |

### Comparison table for the research paper

Fill this table with results only after matched experiments. Use the same maps, tasks,
start positions, robot counts, scenario seeds, and hardware/runtime for each controller.
The current repository implements MAPPO only; **IPPO and rule-based navigation have
not yet been implemented**, so their cells remain empty until you implement and run
those baselines. Include the existing system as an additional column if it is a
distinct controller.

| Metric | Rule-based navigation | IPPO | MAPPO |
|---|---:|---:|---:|
| Successful missions / attempted missions | — | — | — |
| Success rate (%) | — | — | — |
| Mission makespan, successful missions (sim s; mean ± SD) | — | — | — |
| Average task completion time (sim s; mean ± SD) | — | — | — |
| Average robot completion time (sim s; mean ± SD) | — | — | — |
| Actor inference latency (ms; p50 / p95 / p99) | — | — | — |
| Collision episodes / attempted missions | — | — | — |
| Path length (m) | — | — | — |
| Recovery time after defined disturbance (sim s) | — | — | — |

Report the number of scenarios and training seeds, how failed or timed-out missions
were handled, and uncertainty such as confidence intervals. Do not fill the table with
the example calculator values or infer that MAPPO is best from inference speed alone.

### Algorithms to compare

Use the same approved mission plan, maps, robot count, initial poses, tasks, simulator
settings, and evaluation scenario seeds for every controller. Compare:

| Method | What it is | Main question it answers |
|---|---|---|
| Existing system (reference, if available) | Freeze and measure the current mission execution stack | Does the proposed controller improve the actual current system? |
| Rule-based navigation | A documented, deterministic safe baseline (for example, shortest-path planning plus a fixed collision-avoidance rule) | Does learning improve over a transparent non-learning controller? |
| IPPO | PPO with independent per-robot value functions; no centralized critic | Does centralized training in MAPPO help compared with independent RL? |
| MAPPO | Shared/decentralized actors trained with a centralized critic | Does cooperative centralized training improve mission outcomes? |

The three new algorithm candidates are rule-based navigation, IPPO, and MAPPO. Keep
the existing system as an additional reference condition if it is distinct from the
rule-based controller. If they are the same controller, document that and do not count
it twice. Start with one clearly specified rule-based baseline; do not label an
unimplemented controller as an experimental result.

### Metrics and fair timing

Record simulation time and wall-clock time separately. Define mission start after
scenario reset and any agreed warm-up. Record task completion only when the simulator
confirms the task's completion predicate.

- **Mission makespan (sim seconds):** time of the last required task completion minus
  mission start; report only for successful missions.
- **Mean task completion time (sim seconds):** mean of each required task's completion
  timestamp relative to mission start.
- **Mean robot completion time (sim seconds):** mean of robot finish timestamps
  relative to mission start. This is distinct from mean task completion time.
- **Inference latency (wall-clock milliseconds):** time to produce a joint set of
  robot actions from actor observations. Report p50, p95, and p99; warm up first and
  use the same hardware/runtime. Do not include simulator stepping or critic training
  in deployed actor latency.
- **Safety and robustness:** success rate, collision episodes/count, path length, and
  recovery time after a defined disturbance. Report failures/timeouts separately;
  do not treat a timed-out mission as a successful makespan.

Run multiple independent training seeds and evaluate each trained policy on the same
held-out scenario IDs for all methods. Report sample counts, mean and median, and
uncertainty (for example, confidence intervals over paired scenarios). A percent
improvement is meaningful only for matched, successful scenarios:
`100 * (baseline_makespan - candidate_makespan) / baseline_makespan`.

The initial simulator-independent implementation is
[`evaluation/timing.py`](./evaluation/timing.py). The runner should create one
`MissionTimingEvaluator` per episode, call `record_task_completion`,
`record_robot_finish`, and `record_inference_time` from verified simulator events,
then call `result` once the episode terminates. `result` returns `None` for successful
mission-time metrics on failures, while retaining termination elapsed time. The
evaluator does not decide whether a task is complete; the simulator adapter must
provide authoritative completion events.

For actor latency, time one batched decision for all robots with a monotonic wall
clock (for example, `time.perf_counter_ns()`), then pass elapsed milliseconds to
`record_inference_time`. If running inference on a GPU, synchronize the device
immediately before and after the timed call; otherwise asynchronous execution can
under-report latency. Keep simulator stepping outside this timed interval.

### Isaac Sim connection plan

When Isaac Sim is available, implement its adapter under `envs/isaac_sim/`. It should
translate simulator state into the agreed observation contract, send actions through
the simulator's control interface, advance the simulation, and return rewards,
completion events, and termination status. Keep the timing evaluator outside the
algorithm: it should receive timestamped mission/task/robot events and actor-inference
durations from the runner. This lets every controller use the same instrumentation.
Validate the adapter first with reset/step/termination tests and hand-checked
timestamps before running algorithm comparisons.

Suggested order of work:

1. Agree on tasks, completion/success predicates, observations, actions, and robot
   control frequency for the intended simulator.
2. Define the environment interface and extend the current grid world with obstacles,
   task completion, collisions, and deterministic scenario seeds.
3. Add the event-based timing evaluator and tests for success, timeout, task timing,
   and inference timing.
4. Implement and test rule-based and IPPO baselines against MAPPO using the same
   environment interface.
5. Extend the current actor checkpoint support to checkpoint save/load for the full
   policy and add a held-out evaluation runner; export one result row per episode and
   raw task/robot/inference events.
6. Implement the Isaac Sim adapter when the simulator and robot API are available,
   then repeat the matched-scenario evaluation there.

Do not select a winning algorithm from a single run or timing metric. Prefer an
algorithm only if its mission-time improvement is repeatable and does not come with
unacceptable losses in success rate, collisions, or recovery performance.

## Thesis architecture scaffold (`mappo-project/`)

The `mappo-project/` folder is a separate architecture scaffold for the thesis:
**LLM-driven mission planning + MAPPO execution for multi-robot systems, tested in
Isaac Sim**. It does not replace the root-level grid-world prototype documented
above. Its Python modules currently define interfaces and raise
`NotImplementedError`; they are not yet a runnable trainer or simulator integration.

### Architecture and responsibilities

```text
mappo-project/
  configs/       base, Isaac Sim, frozen scenarios, and ablation settings
  envs/          shared environment contract, grid/Isaac adapters, safety filter
  algorithms/    actor, critic, rollout buffer, MAPPO/IPPO trainer
  baselines/     rule-based, random, and existing-system (B0) controller interfaces
  planning/      mission text -> plan -> validation -> plan executor
  evaluation/    seeded runner, timing, inference benchmark, aggregation, plots
  scripts/       multi-seed experiments and ablation orchestration
  integration/   future ROS 2 decision bridge
  tests/         environment, timing, comparison, scenarios, safety, Isaac placeholders
  docs/          metrics, simulator guide, event schema, failure taxonomy, repo notes
  train.py       training entry point interface
  evaluate.py    fair seeded evaluation and report entry point interface
```

The intended flow is:

```text
configs/scenarios.yaml -> frozen, disjoint train/validation/test seed splits
mission text -> LLM planner -> plan validator -> frozen approved plan
train.py / run_experiments.py -> grid or Isaac environment + MAPPO/IPPO -> checkpoints
evaluate.py -> every controller on the same approved plan and test scenarios
             -> run_episode -> timing evaluator -> episodes.csv + events.jsonl
benchmark_inference.py -> actor-only and full-pipeline latency table
logged results -> aggregation -> plots/comparison -> report and final decision
provenance.py -> run_metadata.json beside each result
```

Keep responsibilities separate: environments simulate; algorithms act and learn;
the timing evaluator measures; comparison reads logged rows only. Measure makespan
using simulator-clock time, not reward or wall-clock time. Use the same approved plan,
scenario seeds, and safety rules for every controller in a comparison. Isaac Sim
imports are kept inside `IsaacMultiRobotEnv.__init__`, so importing the scaffold does
not require Isaac Sim.

### Install and check the scaffold (Windows / VS Code terminal)

Open a PowerShell terminal in VS Code. From the repository root, install the
scaffold dependencies into the existing root virtual environment (or create it if
needed):

```powershell
cd "E:\AUC\theis 2\new semaster\MAPPO Agent"
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r .\mappo-project\requirements.txt
```

If `.venv` already exists, skip `python -m venv .venv`. Check that the scaffold
packages import and the Python files compile:

```powershell
cd .\mappo-project
..\.venv\Scripts\python.exe -c "import envs, algorithms, baselines, evaluation, planning; print('Architecture imports OK')"
..\.venv\Scripts\python.exe -m compileall -q envs algorithms baselines evaluation planning scripts integration tests config.py train.py evaluate.py
```

Alternatively, from the repository root, use the full path
`.\.venv\Scripts\python.exe` and run the import check with `sys.path` pointing at
`mappo-project`:

```powershell
cd "E:\AUC\theis 2\new semaster\MAPPO Agent"
.\.venv\Scripts\python.exe -c "import sys; sys.path.insert(0, 'mappo-project'); import envs, algorithms, baselines, evaluation, planning; print('Architecture imports OK')"
```

These are scaffold checks only. `train.py`, `evaluate.py`, and the environment
classes still contain stubs, so training, evaluation, and Isaac Sim execution are
not available yet. The new scaffold test files are also placeholders; they do not
yet validate runtime behavior. For the runnable original grid-world prototype and
its time-metric tools, follow the root-level instructions in
[Install, train, and run the timing tools](#install-train-and-run-the-timing-tools-windows).

### What to do next

Follow the requested build order, completing and testing one step at a time:

1. Implement `envs/grid_env.py` and `evaluation/timing.py`; replace the related test
   placeholders with tests for reproducible resets, the environment contract, and
   simulator-clock metrics.
2. Implement the rule-based baseline, episode runner, comparison utilities, and
   frozen scenario loader. Test that train, validation, and test seeds never overlap.
3. Implement the MAPPO/IPPO algorithm modules and `train.py`, including separate
   validation scenarios and best/last checkpoint saving.
4. Implement evaluation on identical approved plans and test seeds, then aggregation,
   plots, provenance metadata, and the final report.
5. Add and validate the inference benchmark, including warm-up, GPU synchronization,
   hardware/device details, and separate actor-only/full-pipeline latency.
6. Implement action masks, safety overrides with an event for every override, and
   scheduled disturbances.
7. Connect LLM planning, plan validation/freezing, the plan executor, and the
   instrumented existing-system B0 controller.
8. Integrate Isaac Sim when it is available: set the real USD scene path, implement
   seeded reset and the shared step contract, and run the Isaac contract tests.

Do not treat example timestamps, placeholder tests, reward values, or prototype
latency as research results. Preserve scenario/seed matching and record failures,
collisions, success, simulator-time makespan, and run provenance in experiments.
