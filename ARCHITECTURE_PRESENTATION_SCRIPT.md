# Presentation Script: CyberScape and MAPPO Integration

## Suggested 2-minute script

This diagram shows how our existing mission-planning system connects to the MAPPO execution work, and separates components that already exist from the integration work that remains.

The mission starts with an operator describing a task in natural language. CyberScape sends that request to its LLM planner, which produces a structured mission plan. The human approval gate remains in the loop: only an approved plan proceeds to execution.

CyberScape then supplies robot goals or waypoints to a MAPPO integration adapter. This adapter is an important interface we still need to define with the CyberScape and Isaac Sim teams. It must translate goals, robot observations, actions, and execution events without assuming that the MAPPO prototype's current data format matches the robot system.

The robot MAPPO policy uses a decentralized actor: each robot selects actions from its local observation. During training, the centralized critic can use joint state to learn coordinated behavior. That centralized information is for training; deployed robots should act from information available to them through their own sensors and permitted communications.

The policy sends discrete motion actions through the ROS 2 execution and safety layer. That layer maps actions to the robot's timed motion commands and retains the existing safety guards. Commands are executed either in the Isaac Sim digital twin or on TurtleBot3/PuppyPi hardware. Isaac Sim and the robot systems are existing team-owned components; the diagram does not claim that this MAPPO workspace is already connected to them.

Execution outcomes then feed an event-integration path: simulator-clock timestamps, task completion, collisions, and safety-guard interventions. Those events should be connected to the Mission Timing Evaluator so that the team can compare controllers using consistent measures. Disturbances and failures can also be returned to CyberScape to support its adaptive-planning flow, once that feedback contract is agreed and implemented.

The current Python MAPPO workspace is only a grid-world prototype. Its observations describe grid positions and its five actions are grid moves. It is not the LiDAR-based single-robot PPO policy described in the presentation, and its checkpoint cannot directly control a robot. The MAPPO algorithm is a starting point, but the robot observation/action interface, simulator environment, training pipeline, checkpoint, and event integration must be adapted and tested.

For evaluation, we should replay the same approved mission plans, maps, robot starting poses, and scenario seeds across three executors: B0, the existing CyberScape executor; B2, independent PPO policies; and B3, MAPPO. We should report success rate, mission makespan, collisions, idle time, inter-robot conflicts, safety-guard interventions, and policy latency, including new layouts and disturbances. This lets us test coordination, efficiency, and adaptability rather than judging the system from reward curves alone.

The main message is that this thesis is an integration and evaluation effort: CyberScape plans and approves the mission, MAPPO learns coordinated execution, the digital twin provides a repeatable test environment, ROS 2 and safety guards execute commands, and event-based metrics provide evidence. The interfaces between those already-developed systems and the MAPPO policy are the work we need to complete.

## Short version

CyberScape already handles mission planning and human approval, and the team already has the Isaac Sim/ROS 2 environment and single-robot PPO work. Our MAPPO workspace is currently a separate grid-world prototype. The work ahead is to agree on and implement the goal, observation, action, and event contracts; train and evaluate robot-compatible MAPPO in the twin; and compare B0, B2, and B3 on identical missions. The diagram marks those integration tasks explicitly and does not present them as finished.
