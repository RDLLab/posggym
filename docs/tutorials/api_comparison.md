---
layout: "tutorials"
title: API Comparison
firstpage:
---

# API Comparison

Here we give a quick environment API comparison between POSGGym, Gymnasium, and PettingZoo.

## POSGGym

```python
import posggym
env = posggym.make("PredatorPrey-v0")
observations, infos = env.reset(seed=42)

for t in range(100):
    env.render()
    actions = {i: env.action_spaces[i].sample() for i in env.agents}
    observations, rewards, terminations, truncations, all_done, infos = env.step(actions)

    if all_done:
        observations, infos = env.reset()

env.close()
```

## Gymnasium

[Gymnasium API](https://gymnasium.farama.org/)

```python
import gymnasium as gym
env = gym.make("LunarLander-v2", render_mode="human")
observation, info = env.reset(seed=42)

for _ in range(1000):
   action = env.action_space.sample()
   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      observation, info = env.reset()

env.close()
```

## PettingZoo (Parallel API)

[PettingZoo Parallel API](https://pettingzoo.farama.org/api/parallel/)

```python
from pettingzoo.butterfly import pistonball_v6
env = pistonball_v6.parallel_env(render_mode="human")
observations, infos = env.reset(seed=42)

for _ in range(1000):
    actions = {i: env.action_space(i).sample() for i in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)

    if not env.agents:
        observations, infos = env.reset()

env.close()
```
