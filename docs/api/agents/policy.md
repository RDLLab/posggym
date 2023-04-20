# Agent Policy

## posggym.agents.Policy

```{eval-rst}
.. autoclass:: posggym.agents.Policy
```

### Methods

```{eval-rst}
.. autofunction:: posggym.agents.Policy.step
.. autofunction:: posggym.agents.Policy.reset
```

### Attributes

```{eval-rst}
.. autoattribute:: posggym.agents.Policy.model

    The POSG model of the environment (:py:class:`posggym.POSGModel`)

.. autoattribute:: posggym.agents.Policy.agent_id

	The ID of the agent the policy if for.

.. autoattribute:: posggym.agents.Policy.policy_id

	The unique ID of the policy.

.. autoattribute:: posggym.agents.Policy.observes_state

	Whether this policy is requires full observability (i.e. it observes the environment state) or not.

.. autoattribute:: posggym.agents.Policy.spec

    The ``PolicySpec`` of the policy normally set by :py:meth:`posggym.agents.make`

```

### Additional Methods

```{eval-rst}
.. autofunction:: posggym.agents.Policy.get_initial_state
.. autofunction:: posggym.agents.Policy.get_next_state
.. autofunction:: posggym.agents.Policy.sample_action
.. autofunction:: posggym.agents.Policy.get_pi
.. autofunction:: posggym.agents.Policy.get_value
.. autofunction:: posggym.agents.Policy.get_state
.. autofunction:: posggym.agents.Policy.set_state
.. autofunction:: posggym.agents.Policy.get_state_from_history
.. autofunction:: posggym.agents.Policy.close
```
