import os
import os.path as osp
from pathlib import Path


PKG_DIR = osp.dirname(osp.abspath(__file__))
REPO_DIR = osp.abspath(osp.join(PKG_DIR, os.pardir))
BASE_RESULTS_DIR = osp.join(str(Path.home()), "posggym_results")
ASSET_DIR = osp.join(PKG_DIR, "assets")
AGENT_MODEL_DIR = osp.join(ASSET_DIR, "agents")

AGENT_MODEL_REPO_URL = "https://github.com/RDLLab/posggym-agent-models/raw/main/"


if not osp.exists(BASE_RESULTS_DIR):
    os.makedirs(BASE_RESULTS_DIR)
