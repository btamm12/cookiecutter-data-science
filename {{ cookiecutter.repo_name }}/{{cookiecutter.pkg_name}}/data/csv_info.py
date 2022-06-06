from dataclasses import dataclass
from pathlib import Path


@dataclass
class CsvInfo:
    """Utility class that can be used to easily aggregate CSV files with different
    columns.

    To use this class, create a new instance per CSV to parse and specify in which
    columns you can find the required information.

    ```python
    # Information to parse CSV files from dataset A.
    DATASET_A_CSVS = [
        CsvInfo(
            csv_path=DATASET_A_DIR.joinpath(x),
            col_a=0,  # FileName
            col_b=2,  # Label
            col_c=6,  # Other
        ) for x in (
            "real_samples.csv",
            "simulated_samples.csv",
        )
    ]

    # Information to parse CSV files from dataset B.
    DATASET_B_CSVS = [
        CsvInfo(
            csv_path=DATASET_B_DIR.joinpath(x),
            col_a=0,  # FileName
            col_b=1,  # Label
            col_c=2,  # Other
        ) for x in (
            "data.csv",
        )
    ]

    # Construct list of CSVs to be included in the train split.
    TRAIN_CSVS = sum([
        DATASET_A_CSVS,
        DATASET_B_CSVS,
    ], [])
    ```

    Assuming each row of the CSV is a sample, the values can be parsed using the
    following pseudo-code.

    ```python
    out_rows = []
    for row in rows:
        a = row[csv_info.col_a]
        b = row[csv_info.col_b]
        c = row[csv_info.col_c]
        out_rows = [a, b, c]
    ```

    Note that the columns are zero-indexed.
    """

    csv_path: Path
    col_data_path: int  # e.g. input audio path
    col_label: int
    # TODO: Add other important columns or CSV-specific identifiers here, e.g. "for
    # this CSV a transform must be used".


@dataclass
class StandardizedCsvInfo:
    """Contains information about which columns are present in the "standardized"
    CSV.

    This CSV is the result of the preprocessing script and contains all rows
    (samples) of the raw CSVs, but only the relevant columns (and possibly additional
    columns, e.g. feature paths). These columns are organized in a "standardized"
    fashion for all preprocessed CSVs.

    Note that the columns are zero-indexed.
    """

    col_data_path: int = 0  # e.g. input audio path
    col_feature_a_path: int = 1  # e.g. MFCC
    col_feature_b_path: int = 2  # e.g. pre-trained model (XLS-R)
    col_label: int = 3
    col_norm_label: int = 4  # e.g. label normalized in [0,1]
    col_status_label: int = 5  # e.g. label in {0,1} using threshold
    # TODO: Add other important columns here.


# Global instance of "standardized" CSV info.
STANDARDIZED_CSV_INFO = StandardizedCsvInfo()

# TODO: Make sure header is consistent with class definition.
STANDARDIZED_CSV_HEADER = [
    "data_path",
    "feature_a_path",
    "feature_b_path",
    "label",
    "norm_label",
]
