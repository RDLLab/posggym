"""Tests for posggym_agents.rllib.policy module."""
import os
import os.path as osp
import shutil

import posggym

import posggym.agents.grid_world.driving as driving_agents
from posggym.agents.rllib import get_rllib_policy_entry_point
from posggym.agents.utils import download


TEST_ENV_ID = driving_agents.ENV_ID
TEST_ENV_ARGS = {
    "grid": "14x14RoundAbout",
    "num_agents": 2,
    "obs_dim": (3, 1, 1),
    "obstacle_collisions": False,
}
TEST_POLICY_FILE_NAME = "klr_k0_seed0.pkl"
TEST_POLICY_NAME = TEST_POLICY_FILE_NAME.split(".")[0]
TEST_POLICY_VERSION = 0
TEST_POLICY_FILE = osp.join(
    driving_agents.agent_model_dir,
    "driving_14x14roundabout_n2_v0",
    TEST_POLICY_FILE_NAME,
)


def test_load_rllib_policy_spec_with_downloading():
    restore_file = False
    backup_file = ""
    try:
        if osp.exists(TEST_POLICY_FILE):
            # move file so we can restore it later, in case of error
            restore_file = True
            backup_file = TEST_POLICY_FILE + ".bk"
            shutil.move(TEST_POLICY_FILE, backup_file)

        rllib_pi_entry_point = get_rllib_policy_entry_point(TEST_POLICY_FILE)

        env = posggym.make(TEST_ENV_ID, **TEST_ENV_ARGS)
        pi = rllib_pi_entry_point(env.model, env.possible_agents[0], TEST_POLICY_NAME)

        pi.reset()
        obs, _ = env.reset()
        pi.step(obs[env.possible_agents[0]])

        os.remove(TEST_POLICY_FILE)

    finally:
        if restore_file:
            shutil.move(backup_file, TEST_POLICY_FILE)


def test_load_rllib_policy_spec_from_existing_file():
    if not osp.exists(TEST_POLICY_FILE):
        # ensure file is already downloaded
        download.download_from_repo(TEST_POLICY_FILE)

    rllib_pi_entry_point = get_rllib_policy_entry_point(TEST_POLICY_FILE)

    env = posggym.make(TEST_ENV_ID, **TEST_ENV_ARGS)
    pi = rllib_pi_entry_point(env.model, env.possible_agents[0], TEST_POLICY_NAME)

    pi.reset()
    obs, _ = env.reset()
    pi.step(obs[env.possible_agents[0]])


if __name__ == "__main__":
    test_load_rllib_policy_spec_with_downloading()
    test_load_rllib_policy_spec_from_existing_file()
