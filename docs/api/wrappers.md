---
title: Wrapper
---

# Wrappers

```{toctree}
:hidden:
wrappers/misc_wrappers
wrappers/action_wrappers
wrappers/observation_wrappers
wrappers/reward_wrappers
wrappers/petting_zoo
wrappers/rllib
```

POSGGym includes a wrapper API (adopted from [Gymnasium](https://gymnasium.farama.org/api/wrappers/)) as well as a collection of common wrappers that provide a convenient way to modify an existing environment without having to alter the underlying code directly. Wrappers can be applied to any `posggym.Env` environment, and can also be chained to combine their effects.

In order to wrap an environment, you must first initialize a base environment. Then you can pass this environment along with (possibly optional) parameters to the wrapper’s constructor.

```python
>>> import posggym
>>> from posggym.wrappers import FlattenObservation
>>> base_env = posggym.make("PursuitEvasion-v1")
>>> base_env.observation_spaces['0']
Tuple(Tuple(Discrete(2), Discrete(2), Discrete(2), Discrete(2), Discrete(2), Discrete(2)), Tuple(Discrete(16), Discrete(16)), Tuple(Discrete(16), Discrete(16)), Tuple(Discrete(16), Discrete(16)))
>>> wrapped_env = FlattenObservation(base_env)
>>> wrapped_env.observation_spaces['0']
Box(0, 1, (108,), int64)
```

You can access the environment underneath the top-most wrapper by using the `posggym.Wrapper.env` attribute. Since the `posggym.Wrapper` class inherits from `posggym.Env`, the environment from the `posggym.Wrapper.env` attribute can be another wrapper.

```python
>>> wrapped_env
<FlattenObservation<TimeLimit<OrderEnforcing<PassiveEnvChecker<PursuitEvasionEnv<PursuitEvasion-v0>>>>>>
>>> wrapped_env.env
<TimeLimit<OrderEnforcing<PassiveEnvChecker<PursuitEvasionEnv<PursuitEvasion-v0>>>>>
```

If you want to get to the environment underneath all of the layers of wrappers, you can use the `posggym.Wrapper.unwrapped` attribute. If the environment is already a bare environment, this will just return the environment itself.

```python
>>> wrapped_env
<FlattenObservation<TimeLimit<OrderEnforcing<PassiveEnvChecker<PursuitEvasionEnv<PursuitEvasion-v0>>>>>>
>>> wrapped_env.unwrapped
<posggym.envs.grid_world.pursuit_evasion.PursuitEvasionEnv object at 0x7f4a94086d90>
```

There are three common things you might want a wrapper to do:

- Transform actions before applying them to the base environment
- Transform observations that are returned by the base environment
- Transform rewards that are returned by the base environment

Such wrappers can be easily implemented by inheriting from [`posggym.ActionWrapper`](/api/wrappers/action_wrappers), [`posggym.ObservationWrapper`](/api/wrappers/observation_wrappers), or [`posggym.RewardWrapper`](/api/wrappers/reward_wrappers) and implementing the respective transformation. If you need a wrapper to do more complicated tasks, you can inherit from the `posggym.Wrapper` class directly.

## posggym.Wrapper

```{eval-rst}
.. autoclass:: posggym.Wrapper
```

### Methods

```{eval-rst}
.. autofunction:: posggym.Wrapper.step
.. autofunction:: posggym.Wrapper.reset
.. autofunction:: posggym.Wrapper.render
.. autofunction:: posggym.Wrapper.close
```

### Attributes

```{eval-rst}
.. autoproperty:: posggym.Wrapper.model
.. autoproperty:: posggym.Wrapper.state
.. autoproperty:: posggym.Wrapper.possible_agents
.. autoproperty:: posggym.Wrapper.agents
.. autoproperty:: posggym.Wrapper.action_spaces
.. autoproperty:: posggym.Wrapper.observation_spaces
.. autoproperty:: posggym.Wrapper.reward_ranges
.. autoproperty:: posggym.Wrapper.metadata
.. autoproperty:: posggym.Wrapper.spec
.. autoproperty:: posggym.Wrapper.render_mode
.. attribute:: posggym.Wrapper.env

    The environment (one level underneath) this wrapper.

    This may itself be a wrapped environment.
    To obtain the environment underneath all layers of wrappers, use :attr:`posggym.Wrapper.unwrapped`.

.. autoproperty:: posggym.Wrapper.unwrapped
```

## POSGGym Wrappers

POSGGym provides a number of commonly used wrappers listed below.

```{eval-rst}
.. py:currentmodule:: posggym.wrappers

.. list-table::
    :header-rows: 1

    * - Name
      - Type
      - Description
    * - :class:`DiscretizeActions`
      - Action Wrapper
      - An Action wrapper that discretizes continuous action spaces
    * - :class:`RescaleActions`
      - Action Wrapper
      - An Action wrapper for rescaling actions
    * - :class:`FlattenObservations`
      - Observation Wrapper
      - An Observation wrapper that flattens the observation
    * - :class:`RescaleObservations`
      - Observation Wrapper
      - An Observation wrapper for rescaling observations
    * - :class:`OrderEnforcing`
      - Misc Wrapper
      - This will produce an error if `step` or `render` is called before `reset`
    * - :class:`PassiveEnvChecker`
      - Misc Wrapper
      - Checks that the step, reset and render functions follow the posggym API.
    * - :class:`RecordVideo`
      - Misc Wrapper
      - This wrapper will record videos of rollouts.
    * - :class:`TimeLimit`
      - Misc Wrapper
      - This wrapper will emit a truncated signal if the specified number of steps is exceeded in an episode.
```
