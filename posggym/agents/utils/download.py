"""Utility functions for downloading agent files."""
import os
import os.path as osp
import pathlib

import requests
from clint.textui import progress  # type: ignore

from posggym import error, logger
from posggym.config import AGENT_MODEL_REPO_URL

# largest policy file is ~ 1.3-4 MB
LARGEST_FILE_SIZE = int(1.5 * 1024 * 1024)


def download_to_file(url: str, dest_file_path: str):
    """Download file from URL and store at specified destination.

    Arguments
    ---------
    url: full url to download file from
    dest_file_path: file path to write downloaded file to.

    Raises
    ------
    posggym.error.DownloadError: if error occurred while trying to download file.

    """
    dest_dir = osp.dirname(dest_file_path)
    if not osp.exists(dest_dir):
        # create dir if it does not exist
        os.makedirs(dest_dir)

    r = requests.get(url, stream=True)
    if r.ok:
        with open(dest_file_path, "wb") as f:
            content_len = r.headers.get("content-length")
            if isinstance(content_len, str):
                try:
                    total_length = int(content_len)
                except (TypeError,):
                    total_length = LARGEST_FILE_SIZE
            else:
                total_length = LARGEST_FILE_SIZE

            for chunk in progress.bar(
                r.iter_content(chunk_size=1024), expected_size=(total_length / 1024) + 1
            ):
                if chunk:
                    f.write(chunk)
                    f.flush()
                    os.fsync(f.fileno())

    else:
        # HTTP status code 4XX/5XX
        try:
            r.raise_for_status()
        except requests.exceptions.HTTPError as e:
            # wrap exception in posggym-agents error
            raise error.DownloadError(
                f"Error while downloading file, caused by: {type(e).__name__}: {str(e)}"
            ) from e


def download_from_repo(file_path: str, rewrite_existing: bool = False):
    """Download file from the posggym-agent-models github repo.

    Arguments
    ---------
    file_path: local path to posgym package file.
    rewrite_existing: whether to re-download and rewrite an existing copy of the file.

    Raises
    ------
    posggym.error.InvalidFile: if file_path is not a valid posggym-agents package file.
    posggym.error.DownloadError: if error occurred while trying to download file.

    """
    if osp.exists(file_path) and not rewrite_existing:
        return

    path = pathlib.Path(file_path)
    if "agents" not in path.parts:
        raise error.InvalidFile(
            f"Invalid posggym.agents file path '{file_path}'. Path must contain the "
            "`agents` directory."
        )

    base_repo_dir_index = path.parts.index("agents")
    file_repo_url = (
        AGENT_MODEL_REPO_URL + "posggym/" + "/".join(path.parts[base_repo_dir_index:])
    )

    logger.info(
        f"Downloading file from posggym-agent-models repository: {file_repo_url}."
    )

    return download_to_file(file_repo_url, file_path)
