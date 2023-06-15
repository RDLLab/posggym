---
layout: "tutorials"
title: Environment Creation
firstpage:
---

# Creating Custom Environments

This documentation overviews creating new environments and the relevant wrappers, utilities, and tests included in POSGGym that can be useful in the creation of new environments.

Creation of a new environment follows the following high-level steps:

1. Implement the ``POSGModel`` of your custom environment
2. Implement the ``Env`` which essentially wraps your model
3. Add rendering to your ``Env`` (this is optional, but is recommended)
4. Register you new environment

## Example environment ``HurdleRace``

To illustrate the process of implementing a custom environment, we will implement a very simple game ``HurdleRace``. In this environment two agents are racing down a straight path, both agents are in separate tracks, and each track contains hurdles. The first agent to reach the end of their track wins, receiving a reward of `1` while the other agent receives a reward of `-1`. Each agent has two actions: `run` and `jump `. The `run` action moves the agent two cells forward but cannot go over hurdles, while `jump` action moves the agent one cell forward but can go over hurdles. The positions of the hurdles are the same for both agents but are random for each episode. Agents receive a single observation, whether the next cell contains a hurdle or not, so do not receive any information about the other agent or the position of all the hurdles.

The ``HurdleRace`` environment is very simple and really doesn't require any strategy, but will be a nice example for demonstrating how to create a custom environment.

The full code is available at [`posggym/examples/custom_envs/hurdle_race.py`](https://github.com/RDLLab/posggym/blob/main/examples/custom_envs/hurdle_race.py).

## Implementing the model

The first step is to implement the model by subclassing the ``POSGModel``, to better understand what methods are required checkout [the documentation of the POSGModel API](/api/model).

Out custom model will inherit from the abstract class ``posggym.POSGModel`` and will need to implement a number of methods and attributes, including the possible agents, and the action, observation, and (optionally) state space.

> **_NOTE:_**  The implementation below contains type hints. These are not needed for your own implementation, but can be helpful for type checking, etc.

Here is the declaration of the `HurdleRaceModel` and the implementation of `__init__` and some other basic properties of the model:

```python
from __future__ import annotations

from typing import Any, Dict, Tuple, List

from gymnasium import spaces
import numpy as np
import pygame

import posggym
import posggym.utils.seeding as seeding
import posggym.model as M


# The type of an individual states
# This is used for type hinting, and is optional, but encouraged if you plan to share
# your environment with others
HurdleRaceState = Tuple[int, int, int, int, int]


class HurdleRaceModelM.POSGModel[HurdleRaceState, int, int]):
    """The model for the HurdleRace environment."""

    # environment parameters
    TRACK_LENGTH = 10
    N_HURDLES = 3
    JUMP_SUCCESS_RATE = 0.9

    # actions
    RUN = 0
    JUMP = 1

    # observations
    NOHURDLE = 0
    HURDLE = 1

    # rewards
    R_WIN = 1.0
    R_DRAW = 0.0
    R_LOSS = -1.0

    def __init__(self):
        # Tuple of possible agents in our environment
        self.possible_agents = ('0', '1')

		# The state space is actually optional to define, but can be helpful for some
        # algorithms and for debugging.
        # Each state is a tuple containing position of each agent and each hurdle
        # agents can be in position 0, ..., 10 (where 10 is over the finish line)
        # hurdles can be in position 1, ..., 9
        self.state_space = spaces.MultiDiscrete(
            [self.TRACK_LENGTH+1] * 2 + [self.TRACK_LENGTH] * self.N_HURDLES
        )

        # Each agent can either RUN=0 or JUMP=1
        # We create an action space for each agent
        self.action_spaces = {i: spaces.Discrete(2) for i in self.possible_agents}

		# Each agent can observe NOHURDLE=0 or HURDLE=1
        self.observation_spaces = {i: spaces.Discrete(2) for i in self.possible_agents}

		# The environment is symmetric since both agents are identical, only differing
        # by ID
        self.is_symmetric = True

	@property
    def reward_ranges(self) -> Dict[M.AgentID, Tuple[float, float]]:
        # This contains the minimum and maximum reward each agent can receive
        return {i: (self.R_LOSS, self.R_WIN) for i in self.possible_agents}

    @property
    def rng(self) -> seeding.RNG:
        # Here we setup the Random number generator (RNG)
        # See posggym.utils.seeding for some helper functions for using the standard
        # library RNG, and the numpy library RNG.
        # You can also use your own, but for most envs the standard library or numpy
        # library will suffice.
		# If you do use your own, you will also need to overwrite the seed() method.
		# so the rng is seeded appropriately.
        if self._rng is None:
            self._rng, seed = seeding.std_random()
        return self._rng
```

### Get the active agents

Depending on you environment, not all agents may be active at the same time during an episode. For example, an agent may terminate early, or may be added in the middle of an episode. To handle this, the `POSGModel` class defines the `get_agents` which returns the list of IDs of all currently active agents in the environment given the current state. Here we define it for the `HurdleRaceModel`:

```python
    def get_agents(self, state: HurdleRaceState) -> List[M.AgentID]:
        # This is the list of agents active in a given state
        # For our problem both agents are always active, but for some environments
        # agents may leave or join (e.g. via finishing early) and so the active agents
        # can change
        return list(self.possible_agents)
```

### The initial state and observation functions

In the model we need a way of sampling an initial state, and also initial observations for each agent given an initial state. For this purpose, the `POSGModel` class defines the `sample_initial_state` and `sample_initial_obs` methods. These are the methods which will be called to get the initial conditions for an episode. Below is the implementation of these methods for the `HurdleRaceModel`, noting that `self._get_obs(state)` is a helper function that gets the observation given a state and which we will define a bit later.

```python
    def sample_initial_state(self) -> HurdleRaceState:
        # both agents always start at the beginning
        agent_0_pos, agent_1_pos = 0, 0

        # the position of each hurdle is chosen at random, but are spread along the
        # track and are separated by at least one cell
        hurdle_0_pos = self.rng.choice([1, 2])
        hurdle_1_pos = self.rng.choice([4, 5])
        hurdle_2_pos = self.rng.choice([7, 8])

        # the full state
        return (agent_0_pos, agent_1_pos, hurdle_0_pos, hurdle_1_pos, hurdle_2_pos)

    def sample_initial_obs(self, state: HurdleRaceState) -> Dict[M.AgentID, int]:
        # we get the initial observation for an agent (before any action is taken)
        # For this environment the observation is independent of action, so this is easy
        # each agent observes whether the next cell contains a hurdle or not
        # thus we can use our get_obs function
        return self._get_obs(state)
```

### Step

Perhaps the main method for the model is the `step` method. This takes a state and action and returns the next state, observations, rewards, along with terminal and auxiliary information. Below we implement it in the `HurdleRaceModel` along with some helper methods. One thing worth noting is that the model returns an instance of the `posggym.model.JointTimestep` dataclass. This is for convenience since the step method returns so many values, and makes it easier to manage the step methods output.

```python
    def step(
        self, state: HurdleRaceState, actions: Dict[M.AgentID, int]
    ) -> M.JointTimestep[HurdleRaceState, int]:
        # first we get the next state
        next_state = self._get_next_state(state, actions)

		# then the observation given the next state, and joint action
        # In this environment the observation only depends on the state, so we ignore
        # the joint action
        obs = self._get_obs(next_state)

        # Next we get the rewards given the next state
        # Note, technically in a POSG the reward function is defined based on the
        # current state and the action, and next state (or sometimes just the current
        # state and action). In this environment, our reward is based only on the next
        # state, so we ignore the current state, and action.
        # For your own environment you may need to adjust this.
        rewards = self._get_rewards(next_state)

        # An episode is terminated if any agent reaches the end of the track
        end_reached = self.TRACK_LENGTH in next_state[:2]
        terminateds = {i: end_reached for i in self.possible_agents}

        # We leave truncated as False for all agents, this will be handled by the
        # TimeLimit wrapper, if we want there to be a episode step limit
        truncateds = {i: False for i in self.possible_agents}

		# Since all agents are active at all times in our environment, all agents are
        # considered done when any agent reaches the end
        all_done = end_reached

		# Lastly we get the auxiliary info
		infos = self._get_info(next_state)

        # Everything is return in a posggym.model.JointTimestep dataclass object
        # this makes it easier to manage return values of the step function
        return M.JointTimestep(
            next_state, obs, rewards, terminateds, truncateds, all_done, infos
        )

    def _get_next_state(self,
        state: HurdleRaceState, actions: Dict[M.AgentID, int]
    ) -> HurdleRaceState:
        agent_positions = []
        for idx, i in enumerate(self.possible_agents):
            pos = state[idx]
            if actions[i] == self.RUN:
                if not any(pos + 1 == h_pos for h_pos in state[2:]):
                    pos += 1
                    if not any(pos + 1 == h_pos for h_pos in state[2:]):
                        pos += 1
                pos = min(self.TRACK_LENGTH, pos)
            else:
                # JUMP
                if (
                    not any(pos + 1 == h_pos for h_pos in state[2:])
                    or self.rng.random() < self.JUMP_SUCCESS_RATE
                ):
                    pos += 1
            agent_positions.append(pos)
        # the hurdle positions remain unchanged from previous state
        return (
            agent_positions[0], agent_positions[1], *state[2:]
        )

    def _get_obs(self, state: HurdleRaceState) -> Dict[M.AgentID, int]:
        # each agent observes whether the next cell contains a hurdle or not
        obs = {}
        for idx, i in enumerate(self.possible_agents):
            # note the output obs maps agentID to Observation
            # while the state is tuple, so we uses the agent's idx to get their position
            # from the state
            agent_pos = state[idx]
            hurdle_present = any(hurdle_pos == agent_pos+1 for hurdle_pos in state[2:])
            obs[i] = self.HURDLE if hurdle_present else self.NOHURDLE
        return obs

    def _get_rewards(self, state: HurdleRaceState) -> Dict[M.AgentID, float]:
        # agents only receive a reward when at least one agent reaches the end of their
        # track, otherwise the step reward is 0 for both agents
        agent_0_pos, agent_1_pos = state[0], state[1]
        if agent_0_pos == self.TRACK_LENGTH and agent_1_pos == self.TRACK_LENGTH:
            agent_0_reward, agent_1_reward = self.R_DRAW, self.R_DRAW
        elif agent_0_pos == self.TRACK_LENGTH:
            agent_0_reward, agent_1_reward = self.R_WIN, self.R_LOSS
        elif agent_1_pos == self.TRACK_LENGTH:
            agent_0_reward, agent_1_reward = self.R_LOSS, self.R_WIN
        else:
            agent_0_reward, agent_1_reward = 0, 0
        return {"0": agent_0_reward, "1": agent_1_reward}

    def _get_info(self, state: HurdleRaceState) -> Dict[M.AgentID, Dict]:
        # we return the position of the agent each step in the auxiliary information
        # as well as the final outcome
        infos: Dict[M.AgentID, Dict[str, Any]] = {
            i: {"pos": state[idx]} for idx, i in enumerate(self.possible_agents)
        }
        agent_0_pos, agent_1_pos = state[0], state[1]
        if agent_0_pos == self.TRACK_LENGTH and agent_1_pos == self.TRACK_LENGTH:
            infos['0']["outcome"] = M.Outcome.DRAW
            infos['1']["outcome"] = M.Outcome.DRAW
        elif agent_0_pos == self.TRACK_LENGTH:
            infos['0']["outcome"] = M.Outcome.WIN
            infos['1']["outcome"] = M.Outcome.LOSS
        elif agent_1_pos == self.TRACK_LENGTH:
            infos['0']["outcome"] = M.Outcome.LOSS
            infos['1']["outcome"] = M.Outcome.WIN
        return infos
```

And with that, the environment's model is done :). Next we just need to implement the environments `Env` class. Fortunately, we have done a lot of the hard work already.

## Implementing the environment

Our next step is to implement `HurdleRaceEnv` which subclasses the `posggym.Env` abstract class. Fortunately, since we have implemented the majority of the environment logic in the model, the only thing the `HurleRaceEnv` class needs to do is wrap the model and handle the current state, as well as provide rendering functionality. Even more fortunate, is that we can use the `posggym.DefaultEnv` class to manage the model wrapping and handling of state. `posggym.DefaultEnv` implements the `step` and `reset` methods and `state` property of the abstract `posggym.Env` class, so we only need to implement the `render` function, and specify the `metadata`.

Below is the full implementation of the `HurdleRaceEnv` class. In it we initialize our `HurdleRaceModel` then pass it to the parent `posggym.DefaultEnv` class, which then handles the model and state. We then define the `render` function for when `render_mode` is `human` or `rgb_array`. Lastly, we define the `close` method which handles cleanup of any resources.

```python
class HurdleRaceEnv(posggym.DefaultEnv[HurdleRaceState, int, int]):
    """The HurdleRance environment."""

	# Here we specify the meta-data, this should include as a minimum:
    # 'render_modes' - the render modes supported by the environment
    # 'render_fps' - the render framerate to use
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1}

    def __init__(self, render_mode: str | None):
        model = HurdleRaceModel()

        """
        Here we use the posggym.DefaultEnv which implements all the necessary
        functions for and Env given the environment model.
        This will be sufficient for most use cases, or may only require minor tweaks,
        e.g. if the model parameters change between episodes you may need to
        change the reset() function.
        """
        super().__init__(model=model, render_mode=render_mode)

        """
        The main thing we need to implement is environment rendering.

        If human-rendering is used, `self.window` will be a reference to the window that
        we draw to. `self.clock` will be a clock that is used to ensure the environment
        is rendered at the correct framerate in human-mode. The will remain `None` until
        the human render mode is used for the first time.
        """
        self.window_width = 512
        # size of a single cell/pos on the track
        self.cell_size = self.window_width // (model.TRACK_LENGTH + 1)
        self.window_height = self.cell_size * 2
        self.window = None
        self.clock = None

    def render(self):
        if self.render_mode in ("rgb_array", "human"):
            render self._render_frame()
        # render mode is None so do nothing

    def _render_frame(self):
        model: HurdleRaceModel = self.model  # type: ignore

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            pygame.display.set_caption("HurdleRace")
            self.window = pygame.display.set_mode(
                (self.window_width, self.window_height), pygame.SRCALPHA
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_width, self.window_height))
        canvas.fill((255, 255, 255))

        # first we draw the hurdles
        for hurdle_pos in self.state[2:]:
            px_pos = int((hurdle_pos + 0.5) * self.cell_size)
            pygame.draw.line(
                canvas,
                color=(139, 115, 85, 255),  # brown
                start_pos=(px_pos, 0),
                end_pos=(px_pos, 2 * self.cell_size),
                width=self.cell_size // 6,
            )

        # next we draw the agents
        for idx, i in enumerate(self.agents):
            pygame.draw.circle(
                canvas,
                color=(0, 0, 255) if idx == 0 else (255, 0, 0),
                center=(
                    int((self.state[idx] + 0.5) * self.cell_size),
                    int((idx + 0.5) * self.cell_size),
                ),
                radius=self.cell_size // 3,
            )

        # Finally, add some gridlines
        for x in range(model.TRACK_LENGTH + 1):
            pygame.draw.line(
                canvas,
                color=(0, 0, 0),
                start_pos=(0, self.cell_size * x),
                end_pos=(self.window_width, self.cell_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                color=(0, 0, 0),
                start_pos=(self.cell_size * x, 0),
                end_pos=(self.cell_size * x, self.window_height),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate
            # stable.
            self.clock.tick(self.metadata["render_fps"])
        else:
            # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self) -> None:
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
```

## Registering the environment

In order for custom environments to be detected by POSGGym, they must be registered as follows:

```python
from posggym.envs.registration import register
register(
    "posggym_examples/HurdleRace-v0",
    entry_point=HurdleRaceEnv,
    max_episode_steps=50
)
```

POSGGym use the same environment naming conventions as Gymnasium. The environment ID consists of three components, two of which are optional: an optional namespace (here: `posggym_examples`), a mandatory name (here: `HurtleRace`) and an optional but recommended version (here: `v0`). It might have also been registered as `HurtleRace-v0` (the recommended approach), `HurtleRace` or `posggym_examples/HurtleRace`, and the appropriate ID should then be used during environment creation.

The `entry_point` can also be a python path to the class. For example, if `HurdleRaceEnv` could be imported using `from posggym_examples.custom_envs import HurdleRaceEnv` then we could set `entry_point="posggym_examples.custom_envs:HurdleRaceEnv"`, and posggym will handle importing the class.

The keyword argument `max_episode_steps=50` will ensure that HurdleRace environments that are instantiated via `posggym.make` will be wrapped in a `TimeLimit` wrapper [see the wrapper documentation for more information](/api/wrappers). A done signal will then be produced if an agent has reached the end of their track **or** 50 steps have been executed in the current episode. To distinguish truncation and termination, the model outputs the `terminated` and `truncated` values.

## Running the environment

Now that we have implemented our custom `HurdleRace` environment and registered it with `posggym` it's time to take it for a spin. Below we run it using random actions for each agent:

```python

env = posggym.make("posggym_examples/HurdleRace-v0", render_mode="human")
env.reset()

for _ in range(300):
    actions = {i: env.action_spaces[i].sample() for i in env.agents}
    _, _, _, _, all_done, infos = env.step(actions)

    env.render()

    if all_done:
        print("Episode done")
        print(f"Outcome: '0'={infos['0']['outcome']}, '1'={infos['1']['outcome']}")
        env.reset()

env.close()

```

The output should look something like this:

```{figure} ../_static/videos/custom_env/hurdle_race.gif
   :width: 50%
   :align: center

```

<br>

And there you have it, one new custom POSGGym environment.

## More information

* [Env API](/api/env)
* [Model API](/api/model)
* [Wrapper PI](/api/wrappers)
