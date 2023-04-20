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
| MultiAccessBroadcastChannel-v0 | 59370.13         |
| MultiAgentTiger-v0             | 63343.43         |
| RockPaperScissors-v0           | 76395.78         |
| DrivingContinuous-v0           | 551.53           |
| DroneTeamCapture-v0            | 1627.19          |
| PredatorPreyContinuous-v0      | 479.37           |
| PursuitEvasionContinuous-v0    | 481.14           |
| Driving-v0                     | 12128.70         |
| DrivingGen-v0                  | 1997.45          |
| LevelBasedForaging-v2          | 20447.42         |
| PredatorPrey-v0                | 9067.80          |
| PursuitEvasion-v0              | 15427.85         |
| TwoPaths-v0                    | 27268.60         |
| UAV-v0                         | 43672.70         |
