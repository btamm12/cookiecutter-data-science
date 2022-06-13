from pathlib import Path

from {{cookiecutter.pkg_name}} import constants
from {{cookiecutter.pkg_name}}.data.download.utils.scp import scp_dir


def download_eg_dataset_scp():
    """Download the [example] dataset and extract it to the appropriate directory."""

    scp_dir(
        target_ssh_url=constants.EG_DATASET_SSH_URL,
        remote_path=constants.EG_DATASET_REMOTE_DIR,
        local_path=constants.EG_DATASET_LOCAL_DIR,
        jump_ssh_url=None,
        username=None,
        password=None,
        verbose=True,
    )


if __name__ == "__main__":
    download_eg_dataset_scp()
