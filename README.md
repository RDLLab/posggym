# POSGGym

POSGGym is an open source Python library providing implementations of Partially Observable Stochastic Game (POSG) environments coupled with dynamic models of each environment, all under a unified API. While there are a number of amazing open-source implementations for POSG environments, very few have support for dynamic models that can be used for planning. The aim of this library is to fill this gap. Another aim it to provide open-source implementations for many of the environments commonly used in the Partially-Observable multi-agent planning literature. While some open-source implementations exist for some of the common environments, we hope to provide a central repository, with easy to understand and use implementations in order to make reproducibility easier and to aid in faster research.

POSGGym is directly inspired by and adapted from the [Gymnasium (formerly Open AI Gym)](https://gymnasium.farama.org/) and [PettingZoo](https://pettingzoo.farama.org/) libraries for reinforcement learning. The key addition in POSGGym is the support for environment models. POSGGym's API aims to mimic the Gymnasium API as much as possible while incorporating multiple-agents.


## Documentation

The documentation for the project is not yet available online, however you can easily build it yourself :) Head to `docs/README.md` to see how.


## Environments

POSGGym includes the following families of environments. The code for implemented environments are located in the `posggym/envs/` subdirectory.

- *Classic* - These are classic POSG problems from the literature.
- *Grid-World* - These environments are all based in a 2D Gridworld.


## Installation

At the moment we only support installation by cloning the repo and installing locally.

Once the repo is cloned, you can install POSGGym using PIP by navigating to the `posggym` root directory (the one containing the `setup.py` file), and running:

```
pip install -e .
```

Or use the following to install `posggym` with all dependencies:

```
pip install -e .[all]
```


## Environment API

POSGGym models each environment as a python `env` class. Creating environment instances and interacting with them is very simple, and flows almost identically to the Gymnasium user flow. Here's an example using the `TwoPaths-v0` environment:

```python
import posggym
env = posggym.make("TwoPaths-v0")

observations, info = env.reset(seed=42)

for t in range(50):
	actions = {i: env.action_spaces[i].sample() for i in env.agents}
	observations, rewards, terminated, truncated, done, info = env.step(actions)

	if done:
		observation, info = env.reset(seed=42)

env.close()
```


## Model API

Every environment provides access to a model of the environment in the form of a python `model` class. Each model implements a generative model, which can be used for planning, along with functions for sampling initial states. Some environments also implement a full POSG model including the transition, joint observation and joint reward functions.

The following is an example of accessing and using the environment model:


```python
import posggym
env = posggym.make("TwoPaths-v0")
model = env.model

model.seed(seed=42)

state = model.sample_initial_state()
observations = model.sample_initial_obs(state)

for t in range(50):
	actions = {i: env.action_spaces[i].sample() for i in model.get_agents(state)}
	state, observations, rewards, terminated, truncated, all_done, info = model.step(state, actions)

	if all_done:
		state = model.sample_initial_state()
		observations = model.sample_initial_obs(state)
```

The base model API is very similar to the environment API. The key difference that all methods are stateless so can be used repeatedly for planning. Indeed the `env` class for the built-in environments are mainly just a wrappers over the underlying `model` class that manage the state and add support for rendering.

Note that unlike for the `env` class, for convinience the output of the `model.step()` method is a `dataclass` instance and so it's components can be accessed as attributes. For example:

```python
result = model.step(state, actions)
observations = result.observations
info = result.info
```

Both the `env` and `model` classes support a number of other methods, please see the documentation *TODO* for details.


## Authors

**Jonathon Schwartz** - Jonathon.schwartz@anu.edu.au


## License

`MIT` © 2022, Jonathon Schwartz


## Versioning

The POSGGym library uses [semantic versioning](https://semver.org/).
