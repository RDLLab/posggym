"""Tests for the PettingZoo wrapper."""
import pytest

import posggym
from tests.envs.utils import all_testing_env_specs


@pytest.mark.parametrize(
    "spec", all_testing_env_specs, ids=[spec.id for spec in all_testing_env_specs]
)
def test_make_petting_zoo(spec):
    """Checks that posggym -> pettingzoo env conversion works correctly."""
    try:
        from posggym.wrappers.petting_zoo import PettingZoo
        from pettingzoo.utils.conversions import parallel_to_aec_wrapper
        from pettingzoo.test.api_test import api_test
    except ImportError as e:
        pytest.skip(f"pettingzoo not installed.: {str(e)}")

    env = posggym.make(spec.id, disable_env_checker=True)
    pz_env = PettingZoo(env)
    # convert to AEC env so we can use PettingZoo's API test
    aec_env = parallel_to_aec_wrapper(pz_env)
    api_test(aec_env)
