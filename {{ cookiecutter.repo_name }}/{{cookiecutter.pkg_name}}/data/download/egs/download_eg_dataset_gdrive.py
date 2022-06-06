from pathlib import Path

from {{cookiecutter.pkg_name}} import constants
from {{cookiecutter.pkg_name}}.data.download.utils.gdrive import GoogleDriveDownloader
from {{cookiecutter.pkg_name}}.utils.run_once import run_once


def download_eg_dataset_gdrive(
    tmp_dir: Path = None,
):
    """Download the [example] dataset and extract it to the appropriate directory."""

    # Temporary download directory.
    if tmp_dir is None:
        tmp_dir = constants.DIR_DATA.joinpath("tmp")
        tmp_dir.mkdir(mode=0o755, parents=True, exist_ok=True)

    # Create GoogleDriveDownloader object.
    flag_name = "eg_dataset_gdrive_downloaded"
    with run_once(flag_name) as should_run:
        if should_run:
            print("Preparing to download [example] dataset from Google Drive...")
            downloader = GoogleDriveDownloader(constants.GDRIVE_CRED_PATH)
            downloader.download_folder(
                folder_id=constants.EG_DATASET_GDRIVE_ID,
                output_dir=constants.EG_DATASET_GDRIVE_OUTDIR_DIR,
            )

    print("Finished downloading [example] dataset from Google Drive.")


if __name__ == "__main__":
    download_eg_dataset_gdrive()
