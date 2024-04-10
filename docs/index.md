---
hide-toc: true
firstpage:
lastpage:
---

# POSGGym

**POSGGym is a Python library for planning and reinforcement learning research in partially observable, multi-agent environments.**.

```{figure} _static/videos/grid_world/predator_prey.gif
   :alt: Predator Prey
   :width: 300
   :align: center
```

<br/>

POSGGym provides a collection of discrete and continuous environments along with reference agents to allow for reproducible evaluations. The API aims to mimic that of [Gymnasium](https://gymnasium.farama.org/) and [PettingZoo](https://pettingzoo.farama.org/) with the addition of a model API that can be used for planning.

Baseline implementations of planning and reinforcement learning algorithms for POSGGym are available in the [POSGGym-Baselines](https://github.com/RDLLab/posggym-baselines) library. Compatibility with other popular reinforcement learning libraries is possible using the PettingZoo wrapper (see below for an example).


## Environment API

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

## Model API

```python
import posggym
env = posggym.make("PredatorPrey-v0")
model = env.model
model.seed(seed=42)

state = model.sample_initial_state()
observations = model.sample_initial_obs(state)

for t in range(100):
    actions = {i: model.action_spaces[i].sample() for i in model.get_agents(state)}
    state, observations, rewards, terminations, truncations, all_done, infos = model.step(state, actions)

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
env = posggym.make("PursuitEvasion-v1", grid="16x16")

policies = {
    '0': pga.make("PursuitEvasion-v1/grid=16x16/RL1_i0-v0", env.model, '0'),
    '1': pga.make("PursuitEvasion-v1/grid=16x16/ShortestPath-v0", env.model, '1')
}

obs, infos = env.reset(seed=42)
for i, policy in policies.items():
    policy.reset(seed=7)

for t in range(100):
    actions = {i: policies[i].step(obs[i]) for i in env.agents}
    obs, rewards, terminations, truncations, all_done, infos = env.step(actions)

    if all_done:
        obs, infos = env.reset()
        for i, policy in policies.items():
            policy.reset()

env.close()
for policy in policies.values():
    policy.close()
```

(pettingzoo-wrapper)=
## Compatibility with PettingZoo

Any POSGGym environment can be converted into a PettingZoo `ParallelEnv` environment using the `posggym.wrappers.petting_zoo.PettingZoo` wrapper. This allows for easy integration with the ecosystem of libraries that support PettingZoo.

```python
import posggym
from posggym.wrappers.petting_zoo import PettingZoo

env = posggym.make("PredatorPrey-v0")
env = PettingZoo(env)
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
