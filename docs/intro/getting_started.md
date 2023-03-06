---
layout: "contents"
title: Getting Started
firstpage:
---

# Getting Started

POSGGym is a library that provides an API for multi-agent, partially observable reinforcement learning and planning environments represented as a Partially Observable Stochastic Game (POSG) from multi-agent reinforcement learning and planning theory. The thing that distinguishes POSGGym from existing libaries (e.g. [Gymnasium](https://gymnasium.farama.org/) and [PettingZoo](https://pettingzoo.farama.org/) is full support for environment models, which can be used for planning.

The main API of POSGGym is adopted from the Gymnasium and PettingZoo libraries so should be fairly familiar if you have used either of those libaries before.


## Initializing Environments

Initializing environments is very easy in POSGGym and can be done via the ``make`` function (identical to Gymnasium):

```python
import posggym
env = posggym.make('PursuitEvasion-8x8-v0')
```

This will return an ``Env`` for users to interact with. To see all environments you can create, use ``posggym.pprint_registry()``.``make`` includes a number of additional parameters to adding wrappers, specifying keywords to the environment and more.

Furthermore, most environments included in POSGGym can be initialized with different parameters. You can initialize environments with custom parameters either by passing them as keyword arguments to ``make``, or alternatively by loading the environment class directly (similar to PettingZoo).

## Interacting with the Environment

POSGGym models the environment as a Partially Observable Stochastic Game where **all agents act simoultaneously**. So each step all agents perform their action simoultaneously and then all agents recieve their observation and rewards. This is the classic "agent-environment" loop, but differs from multi-agent games like poker or chess where agents act one at a time.

The loop is implemented in posggym using the following code:

```python
import posggym
env = posggym.make("PursuitEvasion-8x8-v0", render_mode="human")
observations, infos = env.reset()

for _ in range(1000):
    actions = {i: env.action_spaces[i].sample() for i in env.agents}
    observations, rewards, terminateds, truncateds, all_done, infos = env.step(actions)

    if all_done:
        observations, infos = env.reset()

env.close()
```

The output should look something like this:

**TODO**

### Explaining the code

First, an environment is created using ``make`` with an additional keyword `"render_mode"` that specifies how the environment should be visualised. See ``render`` for details on the default meaning of different render modes. In this example, we use the ``"PursuitEvasion"`` environment involving two agents: a evader and a pursuer. The evader (red triangle) is trying to reach the goal location (green square) without being spotted by the puruser (blue triangle), while the pursuer is trying to spot the evader before they escape.

After initializing the environment, we ``reset`` the environment to get the first observation for each agent. Note, this is a dictionary where the keys are the ID of each agent, and the values are the observation for each agent. For initializing the environment with a particular random seed or options (see environment documentation for possible values) use the ``seed`` or ``options`` parameters with ``reset``.

Next, agents performs their actions in the environment, ``step``, this can be imagined as multiple robots moving or players pressing a button on a games' controller that causes a change within the environment. As a result, each agent receives a new observation from the updated environment along with their reward for taking the action. Both of these are again returned as dictionaries mapping from agent ID to the agents observation or reward. The reward for each agent could be for instance positive for destroying an enemy or a negative reward for moving into lava. One such action-observation exchange is referred to as a *timestep*.

However, after some timesteps, the environment may end, this is called the terminal state. For instance, in the PursuitEvasion environment the evader may have reached it's goal or been spotted by the pursuer. POSGGym follows the gymnasium API with regards to terminal state; if the environment has terminated, this is returned by ``step``. Similarly, we may also want the environment to end after a fixed number of timesteps, in this case, the environment issues a truncated signal. Note however that in POSGGym there are multiple agents and it's possible that agents reach terminal states at different times, thus the whether an agent has reached a ``terminated`` or ``truncated`` state are returned in a dictionaries, again mapping from agent ID to their terminated or truncated value. In addition to ``terminated`` and ``truncated`` signals for each agent, POSGGym also returns an ``all_done`` signal which is `true` if **all** agents have reached **either** a ``terminated `` or ``truncated`` state. If ``all_done`` is `true`, then ``reset`` should be called next to restart the environment.

> **_NOTE:_**  ``all_done`` is very much a convinience signal, it is possible to determine if an episode is completed using ``terminateds`` and ``truncateds`` returned by the ``step`` function, but for environments where the number of agents can change over time this requires some additional work by the user. The ``all_done`` is one of the main differences between Gymnasium/PettingZoo and POSGGym.


## Agents

Each environment in POSGGym contains multiple-agents acting in a single environment where each agent is identified using a unique string ID. The IDs of all possible agents that can interact with the environment can be accessed using the ``env.possible_agents`` property. Note that depending on the environment not all agents may be active in the environment at one time, or in the same episode. To see the list of currently active agents in the environment use ``env.agents``.

## Action and observation spaces

Every environment specifies the format of valid actions and observations for each agent with the ``env.action_spaces`` and ``env.observation_spaces`` attributes. These attributes map from agent ID to their action or observation spaces. This is helpful for knowing both the expected input and output of the environment for each agent as all valid actions and observation should be contained with the respective space.

In the example, we sampled random actions via ``env.action_space[i].sample()``. In practice actions will be selected using an agent policy, mapping observations to actions.

Every environment should have the attributes ``action_spaces`` and ``observation_spaces``, both of which should be dictionaries whose keys are agentIDs and values are instances of classes that inherit from ``gymnasium.spaces.Space``. For more information about ``Space`` instances, see the [gymnasium documentation](https://gymnasium.farama.org/api/spaces/).

## Interacting with the Model



## Modifying the environment using Wrappers

Coming soon.

## More information

* [Making a Custom environment using the POSGGym API](/tutorials/environment_creation)
