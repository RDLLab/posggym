---
hide-toc: true
firstpage:
lastpage:
---

# POSGGym

**POSGGym is an open source Python library providing implementations of Partially Observable Stochastic Game (POSG) environments coupled with dynamic models of each environment, all under a unified API**.

```{figure} _static/videos/grid_world/predator_prey.gif
   :alt: Predator Prey
   :width: 300
   :align: center
```

<br/>

While there are a number of amazing open-source implementations for POSG environments, very few have support for dynamic models that can be used for planning. The aim of this library is to fill this gap. Another aim it to provide open-source implementations for many of the environments commonly used in the Partially-Observable multi-agent planning literature. While some open-source implementations exist for some of the common environments, we hope to provide a central repository, with easy to understand and use implementations in order to make reproducibility easier and to aid in faster research.

POSGGym is directly inspired by and adapted from the [Gymnasium (formerly Open AI Gym)](https://gymnasium.farama.org/), [PettingZoo](https://pettingzoo.farama.org/), and [Minigrid](https://minigrid.farama.org/) libraries for reinforcement learning. The key addition in POSGGym is the support for environment models which can be used for planning. POSGGym's API aims to stay as close to the Gymnasium API as possible while incorporating multiple-agents into the mix.


## Environment API

```{code-block} python
import posggym

env = posggym.make("TwoPaths-7x7-v0")

observations, info = env.reset(seed=42)

for t in range(50):
    actions = {i: env.action_spaces[i].sample() for i in env.agents}
    observations, rewards, terminated, truncated, done, info = env.step(actions)

    if done:
        observation, info = env.reset(seed=42)

env.close()
```

## Model API

```{code-block} python
import posggym

env = posggym.make("TwoPaths-7x7-v0")
model = env.model

model.seed(seed=42)

state = model.sample_initial_state()
if mode.observation_first:
    observations = model.sample_initial_obs(state)

for t in range(50):
    actions = {i: env.action_spaces[i].sample() for i in model.get_agents(state)}
    state, observations, rewards, terminated, truncated, all_done, info = model.step(state, actions)

    if all_done:
        state = model.sample_initial_state()
        observations = model.sample_initial_obs(state)
```


```{toctree}
:hidden:
:caption: Introduction

intro/getting_started
intro/installation

```

```{toctree}
:hidden:
:caption: API

api/env
api/model
api/wrappers
```

```{toctree}
:hidden:
:caption: Environments

environments/classic
environments/grid_world
```

```{toctree}
:hidden:
:caption: Tutorials

tutorials/environment_creation.md
```

```{toctree}
:hidden:
:caption: Development

Github <https://github.com/RDLLab/posggym>
Contribute to the Docs <https://github.com/RDLLab/posggym/blob/main/docs/README.md>
```
