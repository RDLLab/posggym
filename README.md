[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)



# POSGGym

POSGGym is an open source Python library providing implementations of Partially Observable Stochastic Game (POSG) environments coupled with dynamic models of each environment, all under a unified API. While there are a number of amazing open-source implementations for POSG environments, very few have support for dynamic models that can be used for planning, especially for continuous domains. The aim of this library is to fill this gap.

Another aim it to provide open-source implementations for many of the environments commonly used in the Partially-Observable multi-agent planning literature. While some open-source implementations exist for some of the common environments, we hope to provide a central repository, with easy to understand and use implementations in order to make reproducibility easier and to aid in faster research.

Lastly, another key component of multi-agent research are reference agents that can be used to benchmark new algorithms. POSGGym aims to provide a number of reference agents for each environment, including both hand-coded agents and agents trained using reinforcement learning.

POSGGym is directly inspired by and adapted from the [Gymnasium](https://gymnasium.farama.org/) and [PettingZoo](https://pettingzoo.farama.org/) libraries for reinforcement learning. The key addition in POSGGym is the support for environment models. POSGGym's API aims to stick as close as possible to Gymnasium and PettingZoo Parallel APIs while incorporating models.


## Documentation

The documentation for the project is available at [posggym.readthedocs.io/](https://posggym.readthedocs.io/).

## Installation

We support and test for Python>=3.8.

### Using pip

The latest release version of POSGGym can be installed using `pip`` by running:

```
pip install posggym
```

This will install the base dependencies for running the main environments and download the agent models (so may take a few minutes), but may not include all dependencies for all environments or for rendering some environments, and will not include dependencies for running many posggym agents.

You can install all dependencies for a family of environments like `pip install posggym[grid-world]` and `pip install posggym[continuous]` or dependencies for all environments using `pip install posggym[envs-all]`.

You can install dependencies for POSGGym agents using `pip install posggym[agents]` or to install dependencies for all environments and agents use `pip install posggym[all]`.

### Installing from source

To install POSGGym from source, first clone the repository then run:

```bash
cd posggym
pip install -e .
```

This will install the base dependencies and download the agent models (so may take a few minutes). You can optionally install extras as described above. E.g. to install all dependencies for all environments and agents use:

```bash
pip install -e .[all]
```

## Environments

POSGGym includes the following families of environments (for a full list of environments and their descriptsion see the [documentation](https://posggym.readthedocs.io/)).

- [Classic](https://posggym.readthedocs.io/en/latest/environments/classic.html) - These are classic POSG problems from the literature.
- [Grid-World](https://posggym.readthedocs.io/en/latest/environments/grid_world.html) - These environments are all based in a 2D Gridworld.
- [Continuous](https://posggym.readthedocs.io/en/latest/environments/continuous.html) - 2D environments with continuous state, actions, and observations.


## Environment API

POSGGym models each environment as a python `env` class. Creating environment instances and interacting with them is very simple, and flows almost identically to the Gymnasium user flow. Here's an example using the `PredatorPrey-v0` environment:

```python
import posggym
env = posggym.make("PredatorPrey-v0")
observations, infos = env.reset(seed=42)

for t in range(100):
    actions = {i: env.action_spaces[i].sample() for i in env.agents}
    observations, rewards, terminations, truncations, all_done, infos = env.step(actions)

    if all_done:
        observations, infos = env.reset()

env.close()
```


## Model API

Every environment provides access to a model of the environment in the form of a python `model` class. Each model implements a generative model, which can be used for planning, along with functions for sampling initial states. Some environments also implement a full POSG model including the transition, joint observation and joint reward functions.

The following is an example of accessing and using the environment model:


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

    if all_done:
        state = model.sample_initial_state()
        observations = model.sample_initial_obs(state)
```

The base model API is very similar to the environment API. The key difference that all methods are stateless so can be used repeatedly for planning. Indeed the `env` class for the built-in environments are mainly just a wrappers over the underlying `model` class that manage the state and add support for rendering.

Note that unlike for the `env` class, for convenience the output of the `model.step()` method is a `dataclass` instance and so it's components can be accessed as attributes. For example:

```python
timestep = model.step(state, actions)
observations = timestep.observations
infos = timestep.infos
```

Both the `env` and `model` classes support a number of other methods, please see the documentation [posggym.readthedocs.io/](https://posggym.readthedocs.io/) for more details.

## Agents API

POSGGym.Agents models each agent as a python `policy` class, which at it's simplest accepts an observation and returns the next action. Here's an example using one of the K-Level Reasoning policies in the `PursuitEvasion-v0` environment:


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

For a full explanation of the agent API please see the [POSGGym Agents Getting Started documentation](https://posggym.readthedocs.io/en/latest/agents/getting_started.html). A full list of implemented agents is also available in the documentation.

## Citation

You can cite POSGGym as:

```bibtex
@misc{schwartzPOSGGym2023,
    title = {POSGGym},
    urldate = {2023-08-08},
    author = {Schwartz, Jonathon and Newbury, Rhys and Kurniawati, Hanna},
    year = {2023},
}
```
