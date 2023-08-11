"""Functions and classes for rendering the environment during episodes."""
import abc
from typing import Dict, Iterable

import posggym
import posggym.model as M
from posggym.agents.policy import Policy


class Renderer(abc.ABC):
    """Abstract Renderer Base class."""

    @abc.abstractmethod
    def render_step(
        self,
        episode_t: int,
        env: posggym.Env,
        timestep: M.JointTimestep,
        action: Dict[str, M.ActType],
        policies: Dict[str, Policy],
        episode_end: bool,
    ) -> None:
        """Render a single environment step."""


class EpisodeRenderer(Renderer):
    """Episode Renderer.

    Calls the posggym.Env.render() function.
    """

    def __init__(self, render_frequency: int = 1):
        self._render_frequency = render_frequency
        self._episode_count = 0

    def render_step(
        self,
        episode_t: int,
        env: posggym.Env,
        timestep: M.JointTimestep,
        action: Dict[str, M.ActType],
        policies: Dict[str, Policy],
        episode_end: bool,
    ) -> None:
        if self._episode_count % self._render_frequency != 0:
            self._episode_count += int(episode_end)
            return

        self._episode_count += int(episode_end)
        env.render()


class PauseRenderer(Renderer):
    """Pauses for user input after each step.

    Note, renderers are rendered in order so if you want to pause after all the
    other renderers are done for the step make sure to put the PauseRenderer
    instance at the end of the list of renderers that are passed to the
    generate_renders function, or handle calling each renderer manually.
    """

    def render_step(
        self,
        episode_t: int,
        env: posggym.Env,
        timestep: M.JointTimestep,
        action: Dict[str, M.ActType],
        policies: Dict[str, Policy],
        episode_end: bool,
    ) -> None:
        input("Press ENTER to continue.")


def generate_renders(
    renderers: Iterable[Renderer],
    episode_t: int,
    env: posggym.Env,
    timestep: M.JointTimestep,
    action: Dict[str, M.ActType],
    policies: Dict[str, Policy],
    episode_end: bool,
) -> None:
    """Handle the generation of environment step renderings."""
    for renderer in renderers:
        renderer.render_step(episode_t, env, timestep, action, policies, episode_end)
