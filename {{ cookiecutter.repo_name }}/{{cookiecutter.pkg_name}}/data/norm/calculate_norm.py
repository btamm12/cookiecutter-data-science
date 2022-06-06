import torch
from torch.utils.data import DataLoader

from {{cookiecutter.pkg_name}} import constants
from {{cookiecutter.pkg_name}}.data.csv_dataset import CropType, CsvDataset
from {{cookiecutter.pkg_name}}.data.csv_info import STANDARDIZED_CSV_INFO
from {{cookiecutter.pkg_name}}.data.split import Split, ALL_SPLITS
from {{cookiecutter.pkg_name}}.model.input import ALL_INPUTS
from {{cookiecutter.pkg_name}}.utils.run_once import run_once


def _calculate_norm(split: Split, example: bool):

    # For printing...
    example_str = "(example) " if example else ""
    print(f"{example_str}Calculating norm for {split} set from CSV.")

    # Returns a constants.DatasetDir containing information about the dataset.
    dataset = constants.get_dataset(split, example)

    for input in ALL_INPUTS:

        # Create dataloader.
        csv_dataset = CsvDataset(
            split=split,
            example=example,
            input=input,
            get_label=False,
            do_normalization=False,
            crop_type=CropType.NONE,
        )
        dataloader = DataLoader(csv_dataset, batch_size=None, shuffle=False)

        # Iterate through data and calculate running x and x^2.
        x1 = None
        x2 = None
        N = 0
        for feature, _ in dataloader:
            if feature.dim() == 1:
                feature = feature.unsqueeze(1)
            if feature.dim() != 2:
                raise Exception(
                    f"Expected feature dimensionality of 2 for {input}.")

            if x1 is None:
                x1 = torch.zeros((feature.shape[1]),)
                x2 = torch.zeros((feature.shape[1],))

            # Update running x, x^2 and count.
            x1 += feature.sum(dim=0)
            x2 += feature.pow(2).sum(dim=0)
            N += feature.shape[0]

        # Calculate mean and variance from x, x^2 and N.
        #   μ = 1/N.∑_{i=1..N} (xᵢ)
        #   σ² = 1/N.∑_{i=1..N} (xᵢ²) - μ²
        mu = x1/N
        var = x2/N - mu*mu

        # Save results.
        mu_path = dataset.norm_dir.joinpath(f"{input}.mu.pt")
        torch.save(mu, str(mu_path))
        var_path = dataset.norm_dir.joinpath(f"{input}.var.pt")
        torch.save(var, str(var_path))
        N_path = dataset.norm_dir.joinpath(f"{input}.N.pt")
        torch.save(N, str(N_path))



def calculate_norm(split: Split, example: bool):

    # Flag name. Make sure this operation is only performed once.
    example_name = "_example" if example else ""
    example_str = "(example) " if example else ""
    flag_name = f"calculated_norm_{split}{example_name}"

    # Run exactly once.
    with run_once(flag_name) as should_run:
        if should_run:
            _calculate_norm(split, example)
        else:
            print(f"{example_str}Norm already calculated for {split} split.")


if __name__ == "__main__":
    example: bool = True
    for split in ALL_SPLITS:
        calculate_norm(split, example)
