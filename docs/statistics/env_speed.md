---
layout: "statistics"
title: Environment Speed
firstpage:
---

# Environment Speed

Here we provide some benchmarking results for the speed of each environment in terms of steps per second. These are in no way super accurate numbers since the performance depends on the machine the tests are run on, the configuration of the environment (e.g. problem size, number of agents, etc), and to a lesser extent the behaviour of agents in the environment. Still they can provide some idea of the steps per second that can be expected.

## Benchmarking procedure

We ran each environment with random agents for 10000 steps, and a fixed seed (`42`), with no rendering, and using the default environment configuration. The script used to run the experiment is available in `posggym/scripts/time_env.py`.

## Results

On Intel® Core™ i7-10750H CPU @ 2.60GHz with 16 GB RAM.

| Env                            | Steps per second |
| ------------------------------ | ---------------- |
| MultiAccessBroadcastChannel-v0 | 58095.11         |
| MultiAgentTiger-v0             | 66191.19         |
| RockPaperScissors-v0           | 88039.06         |
| DrivingContinuous-v0           | 565.00           |
| DroneTeamCapture-v0            | 1539.91          |
| PredatorPreyContinuous-v0      | 511.40           |
| PursuitEvasionContinuous-v0    | 477.47           |
| Driving-v0                     | 11943.63         |
| DrivingGen-v0                  | 1925.79          |
| LevelBasedForaging-v2          | 19632.33         |
| PredatorPrey-v0                | 8642.39          |
| PursuitEvasion-v0              | 14946.60         |
| TwoPaths-v0                    | 27706.63         |
| UAV-v0                         | 43118.94         |
