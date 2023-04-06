[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# POSGGym

POSGGym is an open source Python library providing implementations of Partially Observable Stochastic Game (POSG) environments coupled with dynamic models of each environment, all under a unified API. While there are a number of amazing open-source implementations for POSG environments, very few have support for dynamic models that can be used for planning. The aim of this library is to fill this gap. Another aim it to provide open-source implementations for many of the environments commonly used in the Partially-Observable multi-agent planning literature. While some open-source implementations exist for some of the common environments, we hope to provide a central repository, with easy to understand and use implementations in order to make reproducibility easier and to aid in faster research.

POSGGym is directly inspired by and adapted from the [Gymnasium (formerly Open AI Gym)](https://gymnasium.farama.org/) and [PettingZoo](https://pettingzoo.farama.org/) libraries for reinforcement learning. The key addition in POSGGym is the support for environment models. POSGGym's API aims to mimic the Gymnasium API as much as possible while incorporating multiple-agents.

POSGGym also contains a collection of agent policies under the `posggym.agents` sub-module.


## Documentation

The documentation for the project is available at [posggym.readthedocs.io/](https://posggym.readthedocs.io/).


## Installation

The latest version of POSGGym can be installed by running:

```
pip install posggym
```

This will install the base dependencies for running the main environments, but may not include all dependencies for all environments or for rendering some environments, and will not include dependencies for running any in-built posggym agents. You can install all dependencies for a family of environments like `pip install posggym[grid_world]` or dependencies for all environments using `pip install posggym[envs_all]`.

We support and test for Python>=3.8.

### Installing POSGGym Agents

To install dependencies for the agents that come with posggym run:

```
pip install posggym[agents]
```

This will install all dependencies needed to run all the agents.

**Note** this will not download all agent models. These will be downloaded as needed, when the specific model is first initialized. Doing this means only the models used will be downloaded, as opposed to downloading all models which is fairly large (>200 MB).

If you want to download all models at once you can use the provided download script:

```
./download_agents.sh
```

**Note** this only works for Linux/macOS.


## Environments

POSGGym includes the following families of environments. The code for implemented environments are located in the `posggym/envs/` subdirectory.

- *Classic* - These are classic POSG problems from the literature.
- *Grid-World* - These environments are all based in a 2D Gridworld.
- *Continuous* - Continuous state and action adaptation of some grid-world environments.

You can see a list of all environments by running:

```python
import posggym
posggym.pprint_registry()

```


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
        for i, policy in agents.items():
            policy.reset()

env.close()
for policy in policies.values():
    policy.close()
```

In the above code we initialize two of the implemented policies for the `PursuitEvasion-v0` environment by calling the `posggym.agents.make` function and passing in the full policy ID of each policy, the `posggym.Env` environmnent model and the agent ID of the agent the policy will be used for in the environment (this ensures it uses the correct environment properties such as action and observation space).

The policy ID is made up of four parts:

1. `env_id` - the ID of the environment the policy is for: `PursuitEvasion-v0`
2. `env_args_id` - a string representation of the environment arguments used in the version of the environment the policy is for: `grid=16x16`
3. `policy_name` - the name of the policy: `klr_k1_seed0_i0` and `klr_k1_seed0_i1`
4. `version` - the version of the policy: `v0`

The `env_id` and `env_args_id` may be omitted depending on the policy. If the policy is environment agnostic (e.g. the `Random-v0` policy works for any environment) then both the `env_id` and `env_args_id` can be omitted. While if the policy is environment specific, but works for all variations of the environment or the environment has only a single variation (it doesn't have any parameters) then the `env_args_id` can be omitted (e.g. `PursuitEvasion-v0/shortestpath-v0`).

## List of Agents

The project currently has agents implemented for the following POSGGym environments:

- Grid World
  - Driving
  - Level Based Foraging
  - Predator Prey
  - Pursuit Evasion
- Continuous
  - Drone Team Capture

The full list of policies can be obtained using the following code:

```python
import posggym.agents as pga
pga.pprint_registry()
# will display all policies organized by `env_id/env_args_id/`
```

## Authors

**Jonathon Schwartz** - Jonathon.schwartz@anu.edu.au

**Rhys Newbury** - Rhys.newbury@anu.edu.au

## License

`MIT` © 2022, Jonathon Schwartz


## Versioning

The POSGGym library uses [semantic versioning](https://semver.org/).
