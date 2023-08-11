---
title: Env
---

# Env

## posggym.Env

```{eval-rst}
.. autoclass:: posggym.Env
```

### Methods

```{eval-rst}
.. autofunction:: posggym.Env.step
.. autofunction:: posggym.Env.reset
.. autofunction:: posggym.Env.render
.. autofunction:: posggym.Env.close
```

### Attributes

```{eval-rst}
.. autoattribute:: posggym.Env.model

    The underlying POSG model of the environment (:py:class:`posggym.POSGModel`)

.. autoattribute:: posggym.Env.state
.. autoattribute:: posggym.Env.possible_agents
.. autoattribute:: posggym.Env.agents
.. autoattribute:: posggym.Env.action_spaces
.. autoattribute:: posggym.Env.observation_spaces
.. autoattribute:: posggym.Env.reward_ranges
.. autoattribute:: posggym.Env.is_symmetric
.. autoattribute:: posggym.Env.spec

    The ``EnvSpec`` of the environment normally set during :py:meth:`posggym.make`

.. autoattribute:: posggym.Env.metadata

    The metadata of the environment containing rendering modes, rendering fps, etc

.. autoattribute:: posggym.Env.render_mode

    The render mode of the environment determined at initialisation

```


### Additional Methods

```{eval-rst}
.. autoproperty:: posggym.Env.unwrapped
```

## posggym.DefaultEnv

```{eval-rst}
.. autoclass:: posggym.DefaultEnv
```
