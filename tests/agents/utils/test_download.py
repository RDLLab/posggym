"""Tests for the posggym.utils.download utility module."""
import os
import os.path as osp
import shutil

import pytest

from posggym import error
import posggym.agents.grid_world.driving as driving_agents
from posggym.config import AGENT_MODEL_REPO_URL
from posggym.agents.utils import download


TEST_POLICY_FILE_NAME = "klr_k0_seed0.pkl"
TEST_POLICY_FILE = osp.join(
    driving_agents.agent_model_dir,
    "driving_14x14roundabout_n2_v0",
    TEST_POLICY_FILE_NAME,
)
TEST_POLICY_FILE_URL = AGENT_MODEL_REPO_URL + (
    "posggym/agents/grid_world/driving/driving_14x14roundabout_n2_v0/"
    f"{TEST_POLICY_FILE_NAME}"
)
TEST_BAD_POLICY_FILE_URL = AGENT_MODEL_REPO_URL + (
    "posggym/agents/grid_world/not_agents_dir/driving_14x14roundabout_n2_v0/"
    f"{TEST_POLICY_FILE_NAME}"
)
TEST_OUTPUT_DIR = osp.join(osp.dirname(osp.abspath(__file__)), "output")
TEST_FILE_DEST = osp.join(TEST_OUTPUT_DIR, TEST_POLICY_FILE_NAME)


@pytest.fixture(scope="module", autouse=True)
def create_test_output_dir():
    if not osp.exists(TEST_OUTPUT_DIR):
        os.makedirs(TEST_OUTPUT_DIR)

    yield

    if osp.exists(TEST_OUTPUT_DIR):
        shutil.rmtree(TEST_OUTPUT_DIR)


def test_download_to_file():
    download.download_to_file(TEST_POLICY_FILE_URL, TEST_FILE_DEST)
    os.remove(TEST_FILE_DEST)


def test_bad_download_to_file():
    with pytest.raises(error.DownloadError, match="Error while downloading file"):
        download.download_to_file(TEST_BAD_POLICY_FILE_URL, TEST_FILE_DEST)
        if osp.exists(TEST_FILE_DEST):
            # clean-up in case download worked for some reason
            os.remove(TEST_FILE_DEST)


def test_download_from_repo():
    restore_file = False
    backup_file = ""
    try:
        if osp.exists(TEST_POLICY_FILE):
            # copy file so we can restore it later, in case of error
            restore_file = True
            backup_file = TEST_POLICY_FILE + ".bk"
            shutil.copy(TEST_POLICY_FILE, backup_file)

        download.download_from_repo(TEST_POLICY_FILE, rewrite_existing=True)
        os.remove(TEST_POLICY_FILE)

    finally:
        if restore_file:
            shutil.move(backup_file, TEST_POLICY_FILE)


if __name__ == "__main__":
    test_download_to_file()
    test_bad_download_to_file()
    test_download_from_repo()
