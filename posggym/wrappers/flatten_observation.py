from gym import spaces

from posggym import ObservationWrapper, Env


class FlattenObservation(ObservationWrapper):
    """Observation wrapper that flattens the observation."""

    def __init__(self, env: Env):
        super().__init__(env)
        self.observation_spaces = tuple(
            spaces.flatten_space(observation_space)
            for observation_space in env.observation_spaces
        )

    def observations(self, observations):
        return tuple(
            spaces.flatten(self.env.observation_spaces[i], observations[i])
            for i in range(self.env.n_agents)
        )
