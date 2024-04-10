from pathlib import Path


PKG_DIR = Path(__file__).resolve().parent
REPO_DIR = PKG_DIR.parent
BASE_RESULTS_DIR = Path.home() / "posggym_results"
ASSET_DIR = PKG_DIR / "assets"
AGENT_MODEL_DIR = ASSET_DIR / "agents"

AGENT_MODEL_REPO_URL = "https://github.com/RDLLab/posggym-agent-models/raw/main/"

BASE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
