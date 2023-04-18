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
| MultiAccessBroadcastChannel-v0 | 58268.71         |
| MultiAgentTiger-v0             | 66175.21         |
| RockPaperScissors-v0           | 87995.65         |
| DrivingContinuous-v0           | 530.87           |
| DroneTeamCapture-v0            | 1489.70          |
| PredatorPreyContinuous-v0      | 457.22           |
| PursuitEvasionContinuous-v0    | 460.06           |
| Driving-v0                     | 12414.77         |
| DrivingGen-v0                  | 1866.07          |
| LevelBasedForaging-v2          | 18848.95         |
| PredatorPrey-v0                | 8707.06          |
| PursuitEvasion-v0              | 14856.54         |
| TwoPaths-v0                    | 26030.25         |
| UAV-v0                         | 43277.41         |
