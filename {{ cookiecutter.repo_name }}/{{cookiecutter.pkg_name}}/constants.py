from pathlib import Path

from {{cookiecutter.pkg_name}}.data.csv_info import CsvInfo
from {{cookiecutter.pkg_name}}.data.split import Split


# ===== #
# PATHS #
# ===== #

# Locate the root directory of the repository, i.e.
# ```
# {{cookiecutter.repo_name}}
# ```
DIR_REPO = None
for path in Path(__file__).parents:
    if path.name == "{{cookiecutter.repo_name}}":
        DIR_REPO = path
        break
if DIR_REPO is None:
    raise Exception("Unable to locate root dir.")

# {{cookiecutter.repo_name}}/data/
DIR_DATA = DIR_REPO.joinpath("data")
DIR_DATA_FLAGS = DIR_DATA.joinpath("flags")
DIR_DATA_PROCESSED = DIR_DATA.joinpath("processed")
DIR_DATA_RAW = DIR_DATA.joinpath("raw")
DIR_DATA_SUBMISSION = DIR_DATA.joinpath("submission")

# {{cookiecutter.repo_name}}/models/
MODELS_DIR = DIR_REPO.joinpath("models")
XLSR_NAME = "wav2vec2-xls-r-300m"
XLSR_DIR = MODELS_DIR.joinpath(XLSR_NAME)

# {{cookiecutter.repo_name}}/{{cookiecutter.pkg_name}}/
SRC_DIR = DIR_REPO.joinpath("{{cookiecutter.pkg_name}}")

# =============================== #
# DOWNLOAD INFORMATION & RAW DIRS #
# =============================== #

# Google Drive API credentials (see main README for details).
# GDRIVE_CRED_PATH = DIR_REPO.joinpath("gdrive_cred.json")

# CSV info per dataset. Also see CsvInfo:
# ```
# {{cookiecutter.repo_name}}/{{cookiecutter.pkg_name}}/data/csv_info.py
# ````

# DATASET A (uses Google Drive folder ID).
# Columns:
#  - AudioName, Ratings, MOS, AudioType, ScaledMOS
DATASET_A_DEV_DIR = DIR_DATA_RAW.joinpath("dataset_a", "train")
DATASET_A_DEV_DIR.mkdir(mode=0o755, parents=True, exist_ok=True)
DATASET_A_DEV_GDRIVE_ID = "1wIgOqnKA1U-wZQrU8eb67yQyRVOK3SnZ"
DATASET_A_DEV_CSVS = [
    CsvInfo(
        csv_path=DATASET_A_DEV_DIR.joinpath(x),
        col_data_path=0,  # AudioName
        col_label=2,      # MOS
    ) for x in (
        "audio_scaled_mos_cosine.csv",
        "audio_scaled_mos_voices.csv",
    )
]

# DATASET A (uses Google Drive folder ID).
# Columns:
#  - AudioName, Ratings, MOS, AudioType, ScaledMOS
DATASET_A_TEST_DIR = DIR_DATA_RAW.joinpath("dataset_a", "test")
DATASET_A_TEST_DIR.mkdir(mode=0o755, parents=True, exist_ok=True)
DATASET_A_TEST_GDRIVE_ID = "abcdefghijklmnopqrstuvwxyz"
DATASET_A_TEST_CSVS = [
    CsvInfo(
        csv_path=DATASET_A_TEST_DIR.joinpath(x),
        col_data_path=0,  # AudioName
        col_label=2,      # MOS
    ) for x in (
        "data.csv",
    )
]


# DATASET B (uses ZIP file URL).
# Columns:
# - db, con, file, con_description, filename_deg, filename_ref, source, lang,
#   votes, mos, noi, col, dis, loud, noi_std, col_std, dis_std, loud_std,
#   mos_std, filepath_deg, filepath_ref
DATASET_B_DEV_DIR = DIR_DATA_RAW.joinpath("dataset_b", "train")
DATASET_B_DEV_DIR.mkdir(mode=0o755, parents=True, exist_ok=True)
DATASET_B_DEV_URL = "https://zenodo.org/record/4728081/files/NISQA_Corpus.zip"
DATASET_B_DEV_ZIP_FOLDER = "dataset_b_train"  # Extracting ZIP gives this folder.
DATASET_B_DEV_CSVS = [
    CsvInfo(
        csv_path=DATASET_B_DEV_DIR.joinpath(x),
        col_data_path=19,  # filepath_deg
        col_label=9,  # mos
    ) for x in (
        "NISQA_corpus_file.csv",
    )
]


# DATASET B (uses ZIP file URL).
# Columns:
# - db, con, file, con_description, filename_deg, filename_ref, source, lang,
#   votes, mos, noi, col, dis, loud, noi_std, col_std, dis_std, loud_std,
#   mos_std, filepath_deg, filepath_ref
DATASET_B_TEST_DIR = DIR_DATA_RAW.joinpath("dataset_b", "test")
DATASET_B_TEST_DIR.mkdir(mode=0o755, parents=True, exist_ok=True)
DATASET_B_TEST_URL = "https://abcdefghijklmnopqrstuvwxyz/data.zip"
DATASET_B_TEST_ZIP_FOLDER = "dataset_b_train"  # Extracting ZIP gives this folder.
DATASET_B_TEST_CSVS = [
    CsvInfo(
        csv_path=DATASET_B_TEST_DIR.joinpath(x),
        col_data_path=19,  # filepath_deg
        col_label=9,  # mos
    ) for x in (
        "data.csv",
    )
]

# For ease-of-access, concatenate the train and test csvs.
DEV_CSVS = sum([
    DATASET_A_DEV_CSVS,
    DATASET_B_DEV_CSVS,
], [])
TEST_CSVS = sum([
    DATASET_A_TEST_CSVS,
    DATASET_B_TEST_CSVS,
], [])


# ============== #
# PROCESSED DIRS #
# ============== #


class DatasetDir():
    """Structure of each dataset split directory."""

    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self.csv_path = root_dir.joinpath("data.csv")
        self.features_dir = root_dir.joinpath("features")
        self.norm_dir = root_dir.joinpath("norm")
        self.predictions_dir = root_dir.joinpath("predictions")
        self.create_dirs()

    def create_dirs(self):
        self.features_dir.mkdir(mode=0o755, parents=True, exist_ok=True)
        self.norm_dir.mkdir(mode=0o755, parents=True, exist_ok=True)
        self.predictions_dir.mkdir(mode=0o755, parents=True, exist_ok=True)


# TRAIN-VAL SPLIT
VAL_SPLIT = 0.15

# Full datasets.
DATASET_TRAIN = DatasetDir(DIR_DATA_PROCESSED.joinpath("train"))
DATASET_VAL = DatasetDir(DIR_DATA_PROCESSED.joinpath("val"))
DATASET_TEST = DatasetDir(DIR_DATA_PROCESSED.joinpath("test"))

# Example datasets.
DATASET_TRAIN_EG = DatasetDir(DIR_DATA_PROCESSED.joinpath("train_eg"))
DATASET_VAL_EG = DatasetDir(DIR_DATA_PROCESSED.joinpath("val_eg"))
DATASET_TEST_EG = DatasetDir(DIR_DATA_PROCESSED.joinpath("test_eg"))


def get_dataset(split: Split, example: bool):
    if split == Split.TRAIN:
        return DATASET_TRAIN_EG if example else DATASET_TRAIN
    if split == Split.VAL:
        return DATASET_VAL_EG if example else DATASET_VAL
    if split == Split.TEST:
        return DATASET_TEST_EG if example else DATASET_TEST
    raise Exception(f"Unknown split: {split}")
