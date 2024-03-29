---
autogenerated:
title: Level Based Foraging
---

# Level Based Foraging

These policies are for the <a href='../../../environments/grid_world/level_based_foraging'>Level Based Foraging environment</a>. Read environment page for detailed information about the environment.

## Generic
These policies can be used for any version of this environment.



```
env = posggym.make("LevelBasedForaging-v3")
```


| Policy | ID | Valid Agent IDs | Description |
|---|---|---|---|
| `H1` | `LevelBasedForaging-v3/H1-v0` | All | H1 always goes to the closest observed food, irrespective of the foods level. |
| `H2` | `LevelBasedForaging-v3/H2-v0` | All | H2 goes towards the visible food closest to the centre of visible players,     irrespective of food level.      |
| `H3` | `LevelBasedForaging-v3/H3-v0` | All | H3 goes towards the closest visible food with a compatible level. |
| `H4` | `LevelBasedForaging-v3/H4-v0` | All | H4 selects and goes towards the visible food that is furthest from the center of     visible players and that is compatible with the agents level.      |
| `H5` | `LevelBasedForaging-v3/H5-v0` | All | H5 targets a random visible food whose level is compatible with all visible     agents.      |
## num_agents=2-size=10-static_layout=False

```
env = posggym.make(
    "LevelBasedForaging-v3",
    num_agents=2,
    max_agent_level=3,
    size=10,
    max_food=8,
    sight=2,
    force_coop=False,
    static_layout=False,
    observation_mode="tuple"
)
```


| Policy | ID | Valid Agent IDs | Description |
|---|---|---|---|
| `RL1` | `LevelBasedForaging-v3/num_agents=2-size=10-static_layout=False/RL1-v0` | All | Deep RL policy trained using PPO and self-play. |
| `RL2` | `LevelBasedForaging-v3/num_agents=2-size=10-static_layout=False/RL2-v0` | All | Deep RL policy trained using PPO and self-play. |
| `RL3` | `LevelBasedForaging-v3/num_agents=2-size=10-static_layout=False/RL3-v0` | All | Deep RL policy trained using PPO and self-play. |
| `RL4` | `LevelBasedForaging-v3/num_agents=2-size=10-static_layout=False/RL4-v0` | All | Deep RL policy trained using PPO and self-play. |
| `RL5` | `LevelBasedForaging-v3/num_agents=2-size=10-static_layout=False/RL5-v0` | All | Deep RL policy trained using PPO and self-play. |
## num_agents=2-size=10-static_layout=True

```
env = posggym.make(
    "LevelBasedForaging-v3",
    num_agents=2,
    max_agent_level=3,
    size=10,
    max_food=8,
    sight=2,
    force_coop=False,
    static_layout=True,
    observation_mode="tuple"
)
```


| Policy | ID | Valid Agent IDs | Description |
|---|---|---|---|
| `RL1` | `LevelBasedForaging-v3/num_agents=2-size=10-static_layout=True/RL1-v0` | All | Deep RL policy trained using PPO and self-play. |
| `RL2` | `LevelBasedForaging-v3/num_agents=2-size=10-static_layout=True/RL2-v0` | All | Deep RL policy trained using PPO and self-play. |
| `RL3` | `LevelBasedForaging-v3/num_agents=2-size=10-static_layout=True/RL3-v0` | All | Deep RL policy trained using PPO and self-play. |
| `RL4` | `LevelBasedForaging-v3/num_agents=2-size=10-static_layout=True/RL4-v0` | All | Deep RL policy trained using PPO and self-play. |
| `RL5` | `LevelBasedForaging-v3/num_agents=2-size=10-static_layout=True/RL5-v0` | All | Deep RL policy trained using PPO and self-play. |
| `RL6` | `LevelBasedForaging-v3/num_agents=2-size=10-static_layout=True/RL6-v0` | All | Deep RL policy trained using PPO and self-play. |
| `RL7` | `LevelBasedForaging-v3/num_agents=2-size=10-static_layout=True/RL7-v0` | All | Deep RL policy trained using PPO and self-play. |
