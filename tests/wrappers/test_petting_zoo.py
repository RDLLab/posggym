"""Tests for the PettingZoo wrapper."""
import copy

import posggym
import pytest

from tests.envs.utils import all_testing_env_specs


@pytest.mark.parametrize(
    "spec", all_testing_env_specs, ids=[spec.id for spec in all_testing_env_specs]
)
def test_make_petting_zoo(spec):
    """Checks that posggym -> pettingzoo env conversion works correctly."""
    try:
        from pettingzoo.test.api_test import api_test  # type: ignore
        from pettingzoo.utils import agent_selector  # type: ignore
        from pettingzoo.utils.conversions import parallel_to_aec_wrapper  # type: ignore
        from posggym.wrappers.petting_zoo import PettingZoo
    except (ImportError, posggym.error.DependencyNotInstalled) as e:
        pytest.skip(f"pettingzoo not installed.: {str(e)}")

    class custom_parallel_to_aec_wrapper(parallel_to_aec_wrapper):
        """PettingZoo ParallelEnv to AECEnv converter.

        Fixes issue in OG converter which tries to convert actions to integers.
        """

        def step(self, action):
            if (
                self.terminations[self.agent_selection]
                or self.truncations[self.agent_selection]
            ):
                del self._actions[self.agent_selection]
                assert action is None
                self._was_dead_step(action)
                return
            self._actions[self.agent_selection] = action
            if self._agent_selector.is_last():
                obss, rews, terminations, truncations, infos = self.env.step(
                    self._actions
                )

                self._observations = copy.copy(obss)
                self.terminations = copy.copy(terminations)
                self.truncations = copy.copy(truncations)
                self.infos = copy.copy(infos)
                self.rewards = copy.copy(rews)
                self._cumulative_rewards = copy.copy(rews)

                env_agent_set = set(self.env.agents)

                self.agents = self.env.agents + [
                    agent
                    for agent in sorted(self._observations.keys())
                    if agent not in env_agent_set
                ]

                if len(self.env.agents):
                    self._agent_selector = agent_selector(self.env.agents)
                    self.agent_selection = self._agent_selector.reset()

                self._deads_step_first()
            else:
                if self._agent_selector.is_first():
                    self._clear_rewards()

                self.agent_selection = self._agent_selector.next()

    env = posggym.make(spec.id, disable_env_checker=True)
    pz_env = PettingZoo(env)
    # convert to AEC env so we can use PettingZoo's API test
    aec_env = custom_parallel_to_aec_wrapper(pz_env)
    api_test(aec_env)
