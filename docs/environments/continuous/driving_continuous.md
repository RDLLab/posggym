---
autogenerated:
title: Driving Continuous
firstpage:
---

# Driving Continuous

```{figure} ../../_static/videos/continuous/driving_continuous.gif
:width: 200px
:name: driving_continuous
```

This environment is part of the <a href='..'>Continuous environments</a>. Please read that page first for general information.

|   |   |
|---|---|
| Possible Agents | ('0', '1') |
| Action Spaces | {'0': Box([-0.7853982 -0.25     ], [0.7853982 0.25     ], (2,), float32), '1': Box([-0.7853982 -0.25     ], [0.7853982 0.25     ], (2,), float32)} |
| Observation Spaces | {'0': Box([ 0.         0.         0.         0.         0.         0.   0.         0.         0.         0.         0.         0.   0.         0.         0.         0.         0.         0.   0.         0.         0.         0.         0.         0.   0.         0.         0.         0.         0.         0.   0.         0.        -6.2831855 -1.        -1.         0.   0.       ], [ 5.         5.         5.         5.         5.         5.   5.         5.         5.         5.         5.         5.   5.         5.         5.         5.         5.         5.   5.         5.         5.         5.         5.         5.   5.         5.         5.         5.         5.         5.   5.         5.         6.2831855  1.         1.        14.  14.       ], (37,), float32), '1': Box([ 0.         0.         0.         0.         0.         0.   0.         0.         0.         0.         0.         0.   0.         0.         0.         0.         0.         0.   0.         0.         0.         0.         0.         0.   0.         0.         0.         0.         0.         0.   0.         0.        -6.2831855 -1.        -1.         0.   0.       ], [ 5.         5.         5.         5.         5.         5.   5.         5.         5.         5.         5.         5.   5.         5.         5.         5.         5.         5.   5.         5.         5.         5.         5.         5.   5.         5.         5.         5.         5.         5.   5.         5.         6.2831855  1.         1.        14.  14.       ], (37,), float32)} |
| Symmetric | True |
| Import | `posggym.make("DrivingContinuous-v0")` |


The Driving Continuous World Environment.

A general-sum 2D continuous world problem involving multiple agents. Each agent
controls a vehicle and is tasked with driving the vehicle from it's start
location to a destination location while avoiding crashing into other vehicles.
This requires agents to coordinate to avoid collisions and can be used to explore
conventions in multi-agent decision-making.

Possible Agents
---------------
The environment supports two or more agents, depending on the world layout. It is
possible for some agents to finish the episode before other agents by either
crashing or reaching their destination, and so not all agents are guaranteed to be
active at the same time. All agents will be active at the start of the episode.

State Space
-----------
Each state is made up of the state of each vehicle (see `VehicleState` class),
which in turn is defined by the vehicle's:

- `(x, y)` coordinates in [0, world_size]
- direction in [-2π, 2π]
- x, y velocity both in [-1, 1]
- the angular velocity of the vehicle in [-2π, 2π]
- the `(x, y)` coordinate of the vehicles destination
- whether the vehicle has reached it's destination or not: `1` or `0`
- whether the vehicle has crashed or not: `1` or `0`
- the minimum distance to the destination achieved by the vehicle in the current
  episode, if the environment was discrete.

Action Space
------------
Each agent has 2 actions, which are the angular velocity and linear acceleration.
Each agent's actions is made up of two parts. The first action component specifies
the angular velocity in `[-pi/4, pi/4]`, and the second component specifies the
linear acceleration in `[-0.25, 0.25]`.

Observation Space
-----------------
Each agent observes a local circle around themselves as a vector. This is achieved
by a series of 'n_sensors' lines starting at the agent which extend for a distance
of 'obs_dist'. For each line the agent observes the closest entity along the line,
specifically if there is a wall or another vehicle. Along with the sensor reading
each agent also observes their vehicles angle, velocity (in x, y), and the distance
to their destination.

This table enumerates the observation space:

| Index: start          | Description                          |  Values   |
| :-------------------: | :----------------------------------: | :-------: |
| 0                     | Wall distance                        | [0, d]    |
| n_sensors             | Other vehicle distance               | [0, d]    |
| 2 * n_sensors         | Vehicle angle                        | [-2π, 2π] |
| 2 * n_sensors + 1     | Vehicle x velocity                   | [-1, 1]   |
| 2 * n_sensors + 2     | Vehicle y velocity                   | [-1, 1]   |
| 2 * n_sensors + 3     | distance to destination along x axis | [0, s]    |
| 2 * n_sensors + 4     | distance to destination along y axis | [0, s]    |

Where `d = obs_dist` and `s = world.size`

If an entity is not observed by a sensor (i.e. it's not within `obs_dist` or is not
the closest entity to the observing agent along the line), The distance reading will
be `obs_dist`.

The sensor reading ordering is relative to the agent's direction. I.e. the values
for the first sensor at indices `0`, `n_sensors`, correspond to the distance
reading to a wall, and other vehicle, respectively, in the direction the agent is
facing.

Rewards
-------
All agents receive a penalty of `0.0` for each step. They receive a penalty of
`-1.0` for crashing (i.e. hitting another vehicle). A reward of `1.0` is given if
the agent reaches it's destination and a reward of `0.05` is given to the agent at
certain points as it makes progress towards it's destination (i.e. as it reduces
it's minimum distance achieved along the shortest path to the destination for the
episode).

Dynamics
--------
Actions are deterministic and movement is determined by direction the vehicle is
facing and it's speed. Vehicles are able to reverse, but cannot change direction
while reversing.

Max and min velocity are `1.0` and `-1.0`, and max linear acceleration is `0.25`,
while max angular velocity is `π / 4`.

Starting State
--------------
Each agent is randomly assigned to one of the possible starting locations in the
world and one of the possible destination locations, with no two agents starting in
the same location or having the same destination location. The possible start and
destination locations are determined by the world layout being used.

Episodes End
------------
Episodes end when all agents have either reached their destination or crashed. By
default a `max_episode_steps` is also set for each DrivingContinuous environment.
The default value is `200` steps, but this may need to be adjusted when using
larger worlds (this can be done by manually specifying a value for
`max_episode_steps` when creating the environment with `posggym.make`).

Arguments
---------

- `world` - the world layout to use. This can either be a string specifying one of
     the supported worlds, or a custom :class:`DrivingWorld` object
     (default = `"14x14RoundAbout"`).
- `num_agents` - the number of agents in the environment (default = `2`).
- `obs_dist` - the sensor observation distance, specifying the distance away from
     itself which an agent can observe along each sensor (default = `5.0`).
- `n_sensors` - the number of sensor lines eminating from the agent. The agent will
     observe at `n_sensors` equidistance intervals over `[0, 2*pi]`
     (default = `16`).

Available variants
------------------
The DrivingContinuous environment comes with a number of pre-built world layouts
which can be passed as an argument to `posggym.make`, to create different worlds:

| World name        | Max number of agents | World size |
|-------------------|----------------------|----------- |
| `6x6`             | 6                    | 6x6        |
| `7x7Blocks`       | 4                    | 7x7        |
| `7x7CrissCross`   | 6                    | 7x7        |
| `7x7RoundAbout`   | 4                    | 7x7        |
| `14x14Blocks`     | 4                    | 14x14      |
| `14x14CrissCross` | 8                    | 14x14      |
| `14x14RoundAbout` | 4                    | 14x14      |


For example to use the DrivingContinuous environment with the `7x7RoundAbout`
layout and 2 agents, you would use:

```python
import posggym
env = posggym.make('DrivingContinuous-v0', world="7x7RoundAbout", num_agents=2)
```

Version History
---------------
- `v0`: Initial version

References
----------
- Adam Lerer and Alexander Peysakhovich. 2019. Learning Existing Social Conventions
via Observationally Augmented Self-Play. In Proceedings of the 2019 AAAI/ACM
Conference on AI, Ethics, and Society. 107–114.
- Kevin R. McKee, Joel Z. Leibo, Charlie Beattie, and Richard Everett. 2022.
Quantifying the Effects of Environment and Population Diversity in Multi-Agent
Reinforcement Learning. Autonomous Agents and Multi-Agent Systems 36, 1 (2022), 1–16
