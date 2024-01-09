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

While there are a number of amazing open-source implementations for POSG environments, very few have support for dynamic models that can be used for planning, especially for continuous domains. The aim of this library is to fill this gap.

A key goal of POSGGym is to provide easy to use and understand open-source implementations for many of the environments commonly used in the partially observable multi-agent planning literature.

POSGGym also provides a collection of reference agents for it's various environments and a unified Agents API for using these agents. These agents can be used to evaluate algorithms, and includes both hand-coded bots and agents trained using reinforcement learning.

POSGGym is directly inspired by and adapted from the [Gymnasium](https://gymnasium.farama.org/) and [PettingZoo](https://pettingzoo.farama.org/) libraries for reinforcement learning. The key additions in POSGGym are support for environment models which can be used for planning and reference agents. POSGGym's API aims to stay as close to the Gymnasium and PettingZoo Parallel APIs as possible while incorporating models and agents into the mix.

Any POSGGym environment can be converted into a PettingZoo environment using a simple wrapper (`posggym.wrappers.petting_zoo.PettingZoo`). This allows for easy integration with the ecosystem of libraries that support PettingZoo.


## Environment API

```{code-block} python
import posggym
env = posggym.make("PredatorPrey-v0")
observations, info = env.reset(seed=42)

for t in range(50):
    actions = {i: env.action_spaces[i].sample() for i in env.agents}
    observations, rewards, terminations, truncations, all_done, infos = env.step(actions)

    if all_done:
        observations, infos = env.reset()

env.close()
```

## Model API

```{code-block} python
import posggym
env = posggym.make("PredatorPrey-v0")
model = env.model
model.seed(seed=42)

state = model.sample_initial_state()
observations = model.sample_initial_obs(state)

for t in range(50):
    actions = {i: model.action_spaces[i].sample() for i in model.get_agents(state)}
    timestep = model.step(state, actions)

    # timestep attribute can be accessed individually:
    state = timestep.state
    observations = timestep.observations

    # Or unpacked fully
    state, observations, rewards, terminations, truncations, all_done, infos = timestep

    if all_done:
        state = model.sample_initial_state()
        observations = model.sample_initial_obs(state)
```

## Agent API

```python
import posggym
import posggym.agents as pga
env = posggym.make("PursuitEvasion-v0", grid="16x16")

policies = {
    '0': pga.make("PursuitEvasion-v0/grid=16x16/klr_k1_seed0_i0-v0", env.model, '0'),
    '1': pga.make("PursuitEvasion-v0/grid=16x16/klr_k1_seed0_i1-v0", env.model, '1')
}

obs, info = env.reset(seed=42)
for i, policy in policies.items():
    policy.reset(seed=7)

for t in range(100):
    actions = {i: policies[i].step(obs[i]) for i in env.agents}
    obs, rewards, termination, truncated, all_done, info = env.step(actions)

    if all_done:
        obs, info = env.reset()
        for i, policy in policies.items():
            policy.reset()

env.close()
for policy in policies.values():
    policy.close()
```


```{toctree}
:hidden:
:caption: Introduction

intro/getting_started
intro/installation

```

```{toctree}
:hidden:
:caption: Environments

environments/classic
environments/continuous
environments/grid_world
```

```{toctree}
:hidden:
:caption: Agents

agents/getting_started.md
agents/continuous.md
agents/generic.md
agents/grid_world.md
```

```{toctree}
:hidden:
:caption: API

api/env
api/model
api/agents
api/wrappers

```

```{toctree}
:hidden:
:caption: Tutorials

tutorials/environment_creation.md
tutorials/agent_creation.md
tutorials/api_comparison.md
```

```{toctree}
:hidden:
:caption: Examples

examples/index.md
```

```{toctree}
:hidden:
:caption: Statistics

statistics/env_speed.md
```

```{toctree}
:hidden:
:caption: Development

Github <https://github.com/RDLLab/posggym>
release_notes/index.md
Contribute to the Docs <https://github.com/RDLLab/posggym/blob/main/docs/README.md>
```
