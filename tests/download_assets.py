"""A script for testing the download of assets."""
import tempfile
import urllib.request

ASSETS_URL = "https://github.com/RDLLab/posggym-agent-models/tarball/refs/tags/v0.4.0"


# prepare progressbar
def show_progress(block_num, block_size, total_size):
    downloaded_mb = int(block_num * block_size / 1024 / 1024)
    total_mb = int(total_size / 1024 / 1024)
    progress = ((block_num * block_size) / total_size) * 100
    print(
        f"Downloading {progress:.0f}/100 % ({downloaded_mb}/{total_mb} MB)",
        end="\r",
    )


print(f"Downloading assets from {ASSETS_URL}")
tarfile_path = tempfile.mktemp(suffix=".tar.gz")
print(f"Downloading assets to {tarfile_path}")
urllib.request.urlretrieve(ASSETS_URL, filename=tarfile_path, reporthook=show_progress)
