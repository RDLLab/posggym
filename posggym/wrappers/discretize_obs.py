import numpy as np
from gymnasium import spaces

from posggym import ObservationWrapper


class DiscretizeObservations(ObservationWrapper):
    def __init__(self, env, num_bins=10):
        super().__init__(env)

        # Ensure all observation spaces are Box
        assert all(
            isinstance(space, spaces.Box) for space in env.observation_spaces.values()
        ), "All agent observation spaces must be gym.spaces.Box"

        self.num_bins = num_bins
        self._original_obs_spaces = self.observation_spaces
        self._bin_specs = {}

        # Build new observation spaces
        self.observation_spaces = {}
        for agent_id, space in self._original_obs_spaces.items():
            assert space.dtype in (np.float32, np.float64)
            assert np.all(np.isfinite(space.low)) and np.all(np.isfinite(space.high))
            self._bin_specs[agent_id] = {
                "low": space.low,
                "high": space.high,
                "bin_width": (space.high - space.low) / num_bins,
            }
            self.observation_spaces[agent_id] = spaces.MultiDiscrete(
                [num_bins] * space.shape[0]  # type: ignore
            )
        self.model = self._wrap_model(self.model)

    def observations(self, observation):
        return {
            agent_id: self._discretize(obs, self._bin_specs[agent_id])
            for agent_id, obs in observation.items()
        }

    def _wrap_model(self, model):
        class DiscretizedModel:
            def __init__(self, base_model, parent):
                self._model = base_model
                self._parent = parent
                self.observation_spaces = parent.observation_spaces

            def step(self, state, actions):
                step_result = self._model.step(state, actions)

                # Discretize observations in-place
                step_result = step_result._replace(
                    observations=self._parent.observations(step_result.observations)
                )
                return step_result

            def sample_initial_obs(self, state):
                obs = self._model.sample_initial_obs(state)
                return self._parent.observations(obs)

            def __getattr__(self, name):
                return getattr(self._model, name)

        return DiscretizedModel(model, self)

    def _discretize(self, obs, spec):
        obs = np.clip(obs, spec["low"], spec["high"])
        discrete = ((obs - spec["low"]) / spec["bin_width"]).astype(int)
        return tuple(np.clip(discrete, 0, self.num_bins - 1))
