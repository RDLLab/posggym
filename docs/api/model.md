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
.. autofunction:: posggym.POSGModel.get_agents
.. autofunction:: posggym.POSGModel.sample_initial_state
.. autofunction:: posggym.POSGModel.sample_initial_obs
.. autofunction:: posggym.POSGModel.step
.. autofunction:: posggym.POSGModel.seed

```

### Attributes

```{eval-rst}
.. autoattribute:: posggym.POSGModel.possible_agents

	Tuple containing the IDs of all possible agents that can be present in the environment.

.. autoattribute:: posggym.POSGModel.state_space

	The Space object corresponding to all valid states. If implemented, all valid states should be contained within this space.

	Note however, implementing the ``state_space`` attribute is optional as many simulation-based algorithms
    (including RL and MCTS) don't need it to function and the state space be cumbersome to define. In cases where it is not implemented it should be None.

.. autoattribute:: posggym.POSGModel.action_spaces

    A mapping from Agent ID to the Space object corresponding to all valid actions for that agent.

.. autoattribute:: posggym.POSGModel.observation_spaces

	A mapping from Agent ID to the Space object corresponding to all valid observations for that agent.

.. autoattribute:: posggym.POSGModel.reward_ranges
.. autoattribute:: posggym.POSGModel.observation_first

   Whether environment is observation or action first.

   "Observation first" environments start by providing the agents with an observation from the initial belief before any action is taken. Most Reinforcement Learning algorithms typically assume this setting.

    "Action first" environments expect the agents to take an action from the initial belief before providing an observation. Many planning algorithms use this paradigm.

    Note
    ----
	"Action first" environments can always be converted into "Observation first" by introducing a dummy initial observation. Similarly, "Action first" algorithms can be made compatible with "Observation first" environments by introducing a single dummy action for the first step only.

.. autoattribute:: posggym.POSGModel.is_symmetric

	Whether the environment is symmetric or not (is asymmetric).

	An environment is "symmetric" if the ID of an agent in the environment does not affect the agent in anyway (i.e. all agents have the same action and observation spaces, same reward functions, and there are no differences in initial conditions all things considered). Classic examples include Rock-Paper-Scissors, Chess, Poker. In "symmetric" environments the same "policy" should do equally well independent of the ID of the agent the policy is used for.

    If an environment is not "symmetric" then it is "asymmetric", meaning that there are differences in agent properties based on the agent's ID. In "asymmetric" environments there is no guarantee that the same "policy" will work for different agent IDs. Examples include Pursuit-Evasion games, any environments where action and/or observation space differs by agent ID.

.. autoattribute:: posggym.POSGModel.rng

.. autoattribute:: posggym.POSGModel.spec

    The ``EnvSpec`` of the environment normally set during :py:meth:`posggym.make`

```

### Additional Methods

```{eval-rst}
.. autofunction:: posggym.POSGModel.sample_agent_initial_state
```

## posggym.POSGFullModel

```{eval-rst}
.. autoclass:: posggym.POSGFullModel
```

```{eval-rst}
.. automethod:: posggym.POSGFullModel.get_initial_belief
.. automethod:: posggym.POSGFullModel.transition_fn
.. automethod:: posggym.POSGFullModel.observation_fn
.. automethod:: posggym.POSGFullModel.reward_fn
```


## posggym.model.JointTimestep

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

## posggym.model.Outcome

```{eval-rst}
.. autoclass:: posggym.model.Outcome
```
