from gym import spaces

from posggym import ObservationWrapper, Env


class FlattenObservation(ObservationWrapper):
    """Observation wrapper that flattens the observation. """

    def __init__(self, env: Env):
        super().__init__(env)
        self.obs_spaces = tuple(
            spaces.flatten_space(obs_space) for obs_space in env.obs_spaces
        )

    def observations(self, observations):
        return tuple(
            spaces.flatten(self.env.obs_spaces[i], observations[i])
            for i in range(self.env.n_agents)
        )
