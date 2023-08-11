"""Race for glory in HurdleRace!

This file contains an example of a simple custom POSGGym environment, and can be used
as a reference for implementing your own.

You can run the environment from the root repository directory with:

$ python examples/custom_envs/hurdle_race.py

"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import posggym
import posggym.model as M
import posggym.utils.seeding as seeding
from gymnasium import spaces


try:
    import pygame
except ImportError as e:
    raise posggym.error.DependencyNotInstalled(
        "pygame is not installed, run `pip install pygame` or "
        "`pip install posggym[all]`"
    ) from e


# The type of an individual states
# This is used for type hinting, and is optional, but encouraged if you plan to share
# your environment with others
HurdleRaceState = Tuple[int, int, int, int, int]


class HurdleRaceEnv(posggym.DefaultEnv[HurdleRaceState, int, int]):
    """The HurdleRance environment.

    In this environment two agents are racing down a straight path, both agents are in
    separate tracks, and each track contains hurdles. The first agent to reach the end
    of their track wins, receiving a reward of `1` while the other agent receives a
    reward of `-1`. Each agent has two actions: `run` and `jump `. The `run` action
    moves the agent two cells forward but cannot go over hurdles, while `jump` action
    moves the agent one cell forward but can go over hurdles. The positions of the
    hurdles are the same for both agents but are random for each episode. Agents receive
    a single observation, whether the next cell contains a hurdle or not, so do not
    receive any information about the other agent or the position of all the hurdles.

    Agents
    ------
    There are two agents "0" and "1".

    Note, the ID for each agent must be a string, hence the quotes "".

    State
    -----
    Each state is made up of the position of each agent and the position of each hurdle.

    In our version of the environment, the track length is fixed at 10, and there are
    3 hurdles, so the state is a tuple of 5 integers each with a value between [0, 10].

    state = (position "0", position "1", hurdle 0, hurdle 1, hurdle 2)

    Actions
    -------
    Each agent has two actions: RUN=0, JUMP=1

    Observation
    -----------
    Each agent observes whether the next cell in front contains a hurdle or not:
    NOHURDLE=0, HURDLE=1.

    Reward
    ------
    The first agent to reach the end of the track (the winner) receives a reward of +1,
    while the loser receives a reward of -1. If both agents reach the end at the same
    time they both receive a reward of 0.

    Dynamics
    --------
    If an agent uses the RUN action, they will move 2 cells forward or until the next
    hurdle or end of the track, whichever is closest. Using the RUN action in front
    of a hurdle will leave the agent in the same spot.

    If an agent uses the JUMP action, they will move 1 cell forward and can be used to
    move passed hurdles. However, JUMP has a 10% chance of failing when the next cell
    contains a hurdle. If the JUMP fails the agent remains in position (the hurdles are
    solid and fixed in place, unlike a real hurdle race).

    The episode ends when either agent reaches the end of the track position 10.

    """

    # Here we specify the meta-data, this should include as a minimum:
    # 'render_modes' - the render modes supported by the environment
    # 'render_fps' - the render framerate to use
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

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
            return self._render_frame()
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


class HurdleRaceModel(M.POSGModel[HurdleRaceState, int, int]):
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
        # tuple of possible agents in our environment
        self.possible_agents = ("0", "1")
        # The state space is actually optional to define, but can be helpful for some
        # algorithms and for debugging.
        # Each state is a tuple containing position of each agent and each hurdle
        # agents can be in position 0, ..., 10 (where 10 is over the finish line)
        # hurdles can be in position 1, ..., 9
        self.state_space = spaces.MultiDiscrete(
            [self.TRACK_LENGTH + 1] * 2 + [self.TRACK_LENGTH] * self.N_HURDLES
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
    def reward_ranges(self) -> Dict[str, Tuple[float, float]]:
        # This contains the minimum and maximum reward each agent can receive
        return {i: (self.R_LOSS, self.R_WIN) for i in self.possible_agents}

    @property
    def rng(self) -> seeding.RNG:
        # Here we setup the Random number generator (RNG)
        # See posggym.utils.seeding for some helper functions for using the standard
        # library RNG, and the numpy library RNG.
        # You can also use your own, but for most envs the standard library or numpy
        # library will suffice.
        if self._rng is None:
            self._rng, seed = seeding.std_random()
        return self._rng

    def get_agents(self, state: HurdleRaceState) -> List[str]:
        # This is the list of agents active in a given state
        # For our problem both agents are always active, but for some environments
        # agents may leave or join (e.g. via finishing early) and so the active agents
        # can change
        return list(self.possible_agents)

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

    def sample_initial_obs(self, state: HurdleRaceState) -> Dict[str, int]:
        # we get the initial observation for an agent (before any action is taken)
        # For this environment the observation is independent of action, so this is easy
        # each agent observes whether the next cell contains a hurdle or not
        # thus we can use our get_obs function
        return self._get_obs(state)

    def step(
        self, state: HurdleRaceState, actions: Dict[str, int]
    ) -> M.JointTimestep[HurdleRaceState, int]:
        # first we get the next state
        next_state = self._get_next_state(state, actions)
        # we then get the observation given the next state, and joint action
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

    def _get_next_state(
        self, state: HurdleRaceState, actions: Dict[str, int]
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
        return (agent_positions[0], agent_positions[1], *state[2:])

    def _get_obs(self, state: HurdleRaceState) -> Dict[str, int]:
        # each agent observes whether the next cell contains a hurdle or not
        obs = {}
        for idx, i in enumerate(self.possible_agents):
            # note the output obs maps agentID to Observation
            # while the state is tuple, so we uses the agent's idx to get their position
            # from the state
            agent_pos = state[idx]
            hurdle_present = any(
                hurdle_pos == agent_pos + 1 for hurdle_pos in state[2:]
            )
            obs[i] = self.HURDLE if hurdle_present else self.NOHURDLE
        return obs

    def _get_rewards(self, state: HurdleRaceState) -> Dict[str, float]:
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

    def _get_info(self, state: HurdleRaceState) -> Dict[str, Dict]:
        # we return the position of the agent each step in the auxiliary information
        # as well as the final outcome
        infos: Dict[str, Dict[str, Any]] = {
            i: {"pos": state[idx]} for idx, i in enumerate(self.possible_agents)
        }
        agent_0_pos, agent_1_pos = state[0], state[1]
        if agent_0_pos == self.TRACK_LENGTH and agent_1_pos == self.TRACK_LENGTH:
            infos["0"]["outcome"] = M.Outcome.DRAW
            infos["1"]["outcome"] = M.Outcome.DRAW
        elif agent_0_pos == self.TRACK_LENGTH:
            infos["0"]["outcome"] = M.Outcome.WIN
            infos["1"]["outcome"] = M.Outcome.LOSS
        elif agent_1_pos == self.TRACK_LENGTH:
            infos["0"]["outcome"] = M.Outcome.LOSS
            infos["1"]["outcome"] = M.Outcome.WIN
        return infos


def run_hurdle_race():
    # first we register the environment so we can create it using posggym.make
    from posggym.envs.registration import register

    register(
        "posggym_examples/HurdleRace-v0",
        entry_point=HurdleRaceEnv,
        max_episode_steps=50,
    )

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


if __name__ == "__main__":
    # run_hurdle_race()

    import sys

    sys.path.insert(0, "/home/jonathon/code/posggym/docs/scripts")
    from posggym.envs.registration import register

    from gen_gifs import gen_gif

    register(
        "HurdleRace-v0",
        entry_point=HurdleRaceEnv,
        max_episode_steps=50,
    )

    gen_gif("HurdleRace-v0", custom_env=True, ignore_existing=True, resize=False)
