import csv
import math

from {{cookiecutter.pkg_name}} import constants
from {{cookiecutter.pkg_name}}.data.csv_info import STANDARDIZED_CSV_HEADER
from {{cookiecutter.pkg_name}}.data.split import Split, ALL_SPLITS
from {{cookiecutter.pkg_name}}.utils.run_once import run_once



def _create_example_csv(split: Split):

    # Returns a constants.DatasetDir containing information about the dataset.
    dataset = constants.get_dataset(split, example=True)

    # Print split name.
    print(f"Creating example CSV for split: {split}.")

    # Load CSV for this split.
    rows = []
    with open(dataset.csv_path, encoding="utf8", mode="r") as in_csv:
        csv_reader = csv.reader(in_csv)
        for in_row in csv_reader:
            rows.append(in_row)

    # Remove header row.
    rows.pop(0)

    # Create example CSV by keeping top X %.
    rows_to_keep = math.ceil(constants.EXAMPLE_FRAC * len(rows))
    assert rows_to_keep > 0 and rows_to_keep <= len(rows)
    rows = rows[:rows_to_keep]

    # Train/val split.
    val_rows = math.ceil(constants.VAL_SPLIT * len(rows))
    train_rows = len(rows) - val_rows
    if split == Split.TRAIN:
        rows = rows[:train_rows]
    if split == Split.VAL:
        rows = rows[train_rows:]

    # Add header row.
    rows.insert(0, STANDARDIZED_CSV_HEADER)

    # Write to output CSV.
    with open(dataset.csv_path, mode="w", encoding="utf8") as f_out:
        csv_writer = csv.writer(f_out)
        csv_writer.writerows(rows)

    print(f"Finished.")


def create_example_csv(split: Split):

    # Flag name. Make sure this operation is only performed once.
    flag_name = f"created_example_csv_{split}"

    # Run exactly once.
    with run_once(flag_name) as should_run:
        if should_run:
            _create_example_csv(split)
        else:
            print(f"Example CSV already created for {split} split.")


if __name__ == "__main__":
    for split in ALL_SPLITS:
        create_example_csv(split)
