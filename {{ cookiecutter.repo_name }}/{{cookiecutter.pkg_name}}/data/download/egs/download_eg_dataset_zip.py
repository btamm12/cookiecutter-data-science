from pathlib import Path

from {{cookiecutter.pkg_name}} import constants
from {{cookiecutter.pkg_name}}.data.download.utils.download_dataset_zip import download_dataset_zip


def download_eg_dataset_zip(
    tmp_dir: Path = None,
    tqdm_name: str = None,
    tqdm_idx: int = None,
):
    """Download the [example] dataset and extract it to the appropriate directory."""

    download_dataset_zip(
        name="example",
        data_url=constants.EG_DATASET_ZIP_URL,
        output_dir=constants.EG_DATASET_ZIP_DIR,
        extracted_name=constants.EG_DATASET_ZIP_FOLDER,
        tmp_dir=tmp_dir,
        tqdm_name=tqdm_name,
        tqdm_idx=tqdm_idx,
    )


if __name__ == "__main__":
    download_eg_dataset_zip(tqdm_name="example", tqdm_idx=0)
