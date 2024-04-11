[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


# POSGGym

POSGGym is a Python library for planning and reinforcement learning research in partially observable, multi-agent environments. It provides a collection of discrete and continuous environments along with reference agents to allow for reproducible evaluations. The API aims to mimic that of [Gymnasium](https://gymnasium.farama.org/) and [PettingZoo](https://pettingzoo.farama.org/) with the addition of a model API that can be used for planning.

The documentation for the project is available online at [posggym.readthedocs.io/](https://posggym.readthedocs.io/). For a guide to building the documentation locally see [docs/README.md](docs/README.md).

Some baseline implementations of planning and reinforcement learning algorithms for POSGGym are available in the [POSGGym-Baselines](https://github.com/RDLLab/posggym-baselines) library. Compatibility with other popular reinforcement learning libraries is possible using the [PettingZoo wrapper](#compatibility-with-pettingzoo).

## Installation

POSGGym supports and tests for `Python>=3.8`. We recommend using a virtual environment to install POSGGym (e.g. [conda](https://docs.conda.io/projects/conda/en/latest/index.html), [venv](https://docs.python.org/3/library/venv.html)).

### Using pip

The latest release version of POSGGym can be installed using `pip` by running:

```bash
pip install posggym
```

This will install the base dependencies for running all the environments and download the agent models (so may take a few minutes). In order to minimise the number of unused dependencies installed the default install does not include dependencies for running many posggym agents (specifically PyTorch).

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

To run tests, install the test dependencies and then run the tests:

```bash
pip install -e .[testing]
pytest
```

Or alternatively you can run one of the examples from the `examples` directory:

```bash
python examples/run_random_agents.py --env_id Driving-v1 --num_episodes 10 --render_mode human
```

## Environments

POSGGym includes the following families of environments (for a full list of environments and their descriptsion see the [documentation](https://posggym.readthedocs.io/)).

- [Classic](https://posggym.readthedocs.io/en/latest/environments/classic.html) - These are classic POSG problems from the literature.
- [Grid-World](https://posggym.readthedocs.io/en/latest/environments/grid_world.html) - These environments are all based in a 2D Gridworld.
- [Continuous](https://posggym.readthedocs.io/en/latest/environments/continuous.html) - 2D environments with continuous state, actions, and observations.


## Environment API

POSGGym models each environment as a python `Env` class. Creating environment instances and interacting with them is very simple, and flows almost identically to the Gymnasium user flow. Here's an example using the `PredatorPrey-v0` environment:

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

Every environment provides access to a model of the environment in the form of a `POSGModel` class. Each model implements a generative model, which can be used for planning, along with functions for sampling initial states. Some environments also implement a full POSG model including the transition, joint observation and joint reward functions.

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

The base model API is very similar to the environment API. The key difference that all methods are stateless so can be repeatedly sampled for planning. Indeed the `Env` class implementations for the built-in environments are a wrapper over an underlying `POSGModel` class that manages the state and adds support for rendering.

Note that unlike for `Env` class, for convenience the output of the `model.step()` method is a `dataclass` instance and so it's components can be accessed as attributes. For example:

```python
timestep = model.step(state, actions)
observations = timestep.observations
infos = timestep.infos
```

Both the `Env` and `POSGModel` classes support a number of additional methods, refer to the [documentation](https://posggym.readthedocs.io/) for more details.

## Agents API

The Agents API provides a way to easy load reference policies that come with POSGGym. Each policy is a `Policy` class, which at it's simplest accepts an observation and returns the next action. The basic Agents API is shown below:


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

For a full explanation of the agent API please see the [POSGGym Agents Getting Started documentation](https://posggym.readthedocs.io/en/latest/agents/getting_started.html). A full list of implemented agents is also available in the documentation.

## Compatibility with PettingZoo

Any POSGGym environment can be converted into a PettingZoo `ParallelEnv` environment using the `posggym.wrappers.petting_zoo.PettingZoo` wrapper. This allows for easy integration with the ecosystem of libraries that support PettingZoo.

```python
import posggym
from posggym.wrappers.petting_zoo import PettingZoo

env = posggym.make("PredatorPrey-v0")
env = PettingZoo(env)
```

## Citation

You can cite POSGGym as:

```bibtex
@misc{schwartzPOSGGym2023,
    title = {POSGGym},
    urldate = {2023-08-08},
    author = {Schwartz, Jonathon and Newbury, Rhys and Kuli\'{c}, Dana and Kurniawati, Hanna},
    year = {2023},
}
```
