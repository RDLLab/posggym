"""Setup for the posggym package."""
import os
import pathlib
import shutil
import tarfile
import urllib.request

from setuptools import setup
from setuptools.command import build_py


CWD = pathlib.Path(__file__).absolute().parent

ASSETS_URL = "https://github.com/RDLLab/posggym-agent-models/tarball/refs/tags/v0.4.0"


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
        tar_file_path = os.path.join(self.get_package_dir("assets"), "assets.tar.gz")
        if os.path.exists(tar_file_path):
            print(f"Found cached assets {tar_file_path}", flush=True)
        else:
            os.makedirs(os.path.dirname(tar_file_path), exist_ok=True)
            print("Downloading assets...", flush=True)
            urllib.request.urlretrieve(
                ASSETS_URL, filename=tar_file_path, reporthook=show_progress
            )
            print(f"Downloaded assets to {tar_file_path}", flush=True)

        root_dir = os.path.join(self.get_package_dir(""), "posggym")
        asset_dir = os.path.join(root_dir, "assets")
        os.makedirs(root_dir, exist_ok=True)
        if os.path.exists(asset_dir):
            shutil.rmtree(asset_dir)
            print("Deleted existing assets", flush=True)

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

    def build_assets(self):
        """Copies assets from package to build directory."""
        print("Building assets...", flush=True)
        package_root = os.path.join(self.get_package_dir(""), "posggym")
        os.makedirs(package_root, exist_ok=True)
        build_root = os.path.join(self.build_lib, "posggym")

        if os.path.exists(f"{build_root}/assets"):
            shutil.rmtree(f"{build_root}/assets")
            print("deleted existing assets", flush=True)

        shutil.copytree(f"{package_root}/assets", f"{build_root}/assets")
        print(
            f"copied assets from {package_root}/assets to {build_root}/assets",
            flush=True,
        )


setup(name="posggym", version=get_version(), cmdclass={"build_py": BuildPy})
