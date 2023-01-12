---
title: Model
---

# Model

## posggym.Model

```{eval-rst}
.. autoclass:: posggym.POSGModel
```

### Methods

```{eval-rst}
.. autofunction:: posggym.POSGModel.sample_initial_state
.. autofunction:: posggym.POSGModel.sample_initial_obs
.. autofunction:: posggym.POSGModel.step
.. autofunction:: posggym.POSGModel.get_agents
.. autofunction:: posggym.POSGModel.seed

```

### Attributes

```{eval-rst}
.. autoattribute:: posggym.POSGModel.model

    The POSG model of the environment (:py:class:`posggym.POSGModel`)

.. autoattribute:: posggym.POSGModel.state
.. autoattribute:: posggym.POSGModel.possible_agents
.. autoattribute:: posggym.POSGModel.state_space
.. autoattribute:: posggym.POSGModel.action_spaces
.. autoattribute:: posggym.POSGModel.observation_spaces
.. autoattribute:: posggym.POSGModel.reward_ranges
.. autoattribute:: posggym.POSGModel.observation_first
.. autoattribute:: posggym.POSGModel.is_symmetric
.. autoattribute:: posggym.POSGModel.spec
mod
    The ``EnvSpec`` of the environment normally set during :py:meth:`posggym.make`

.. autoattribute:: posggym.POSGModel.render_mode

    The render mode of the environment determined at initialisation

```

### Additional Methods

```{eval-rst}
.. autofunction:: posggym.POSGModel.sample_initial_agent_state
```

## posggym.JointTimestep

```{eval-rst}
.. autoclass:: posggym.model.JointTimestep
```

```{eval-rst}
.. autoattribute:: posggym.model.JointTimestep.state
.. autoattribute:: posggym.model.JointTimestep.observations
.. autoattribute:: posggym.model.JointTimestep.rewards
.. autoattribute:: posggym.model.JointTimestep.terminated
.. autoattribute:: posggym.model.JointTimestep.truncated
.. autoattribute:: posggym.model.JointTimestep.all_done
.. autoattribute:: posggym.model.JointTimestep.info
```
