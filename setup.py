"""Setup for the posggym package."""

import shutil
import tarfile
import urllib.request
from pathlib import Path

from setuptools import setup
from setuptools.command import build_py

CWD = Path(__file__).absolute().parent

ASSETS_URL = (
    "https://github.com/RDLLab/posggym-agent-models/archive/refs/heads/main.tar.gz"
)


def get_version():
    """Gets the posggym version."""
    path = CWD / "posggym" / "__init__.py"
    content = path.read_text()

    for line in content.splitlines():
        if line.startswith("__version__"):
            return line.strip().split()[-1].strip().strip('"')
    raise RuntimeError("bad version data in __init__.py")


def show_progress(block_num, block_size, total_size):
    downloaded_mb = int(block_num * block_size / 1024 / 1024)
    total_mb = int(total_size / 1024 / 1024)
    progress = int((block_num * block_size) / total_size) * 100
    print(
        f"Downloading {progress}/100 % ({downloaded_mb}/{total_mb} MB)",
        end="\r",
    )


class BuildPy(build_py.build_py):
    """Command that downloads POSGGym assets as part of build_py."""

    def run(self):
        self.download_and_extract_assets()
        if not self.editable_mode:
            super().run()
            self.build_assets()

    def download_and_extract_assets(self):
        """Downloads and extracts the assets."""
        print("Downloading and extracting assets...", flush=True)
        tar_file_path = self.get_package_dir("assets") / "assets.tar.gz"
        if tar_file_path.exists():
            print(f"Found cached assets {tar_file_path}", flush=True)
        else:
            tar_file_path.parent.mkdir(exist_ok=True)
            print("Downloading assets...", flush=True)
            urllib.request.urlretrieve(
                ASSETS_URL, filename=tar_file_path, reporthook=show_progress
            )
            print(f"Downloaded assets to {tar_file_path}", flush=True)

        root_dir = self.get_package_dir("") / "posggym"
        asset_dir = root_dir / "assets"
        root_dir.mkdir(exist_ok=True)
        if asset_dir.exists():
            shutil.rmtree(asset_dir)
            print("Deleted existing assets", flush=True)
        asset_dir.mkdir(exist_ok=True)
        print("Extracting assets...", flush=True)

        def members(tarball):
            # strip the top-level directory from the tarball
            # ref:
            # https://stackoverflow.com/questions/8008829/extract-only-a-single-directory-from-tar-in-python
            top_level_dirs = "/".join(tarball.getmembers()[-1].name.split("/")[:2])
            for member in tarball.getmembers():
                if member.path.startswith(top_level_dirs):
                    member.path = member.path[len(top_level_dirs) + 1 :]
                    yield member

        with tarfile.open(tar_file_path, mode="r") as tarball:
            tarball.extractall(asset_dir, members=members(tarball))
        print(f"Extracted assets from {tar_file_path} to {asset_dir}", flush=True)

    def get_package_dir(self, package: str) -> Path:
        """Gets the package directory."""
        return Path(super().get_package_dir(package))

    def build_assets(self):
        """Copies assets from package to build directory."""
        print("Building assets...", flush=True)
        package_root = self.get_package_dir("") / "posggym"
        package_root.mkdir(exist_ok=True)
        build_root = Path(self.build_lib) / "posggym"

        if (build_root / "assets").exists():
            shutil.rmtree(build_root / "assets")
            print("deleted existing assets", flush=True)

        shutil.copytree(package_root / "assets", build_root / "assets")
        print(
            f"copied assets from {package_root}/assets to {build_root}/assets",
            flush=True,
        )


setup(name="posggym", version=get_version(), cmdclass={"build_py": BuildPy})
