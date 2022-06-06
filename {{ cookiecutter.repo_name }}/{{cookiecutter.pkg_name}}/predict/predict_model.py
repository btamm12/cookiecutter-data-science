import os
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from typing import List

from {{cookiecutter.pkg_name}} import constants
from {{cookiecutter.pkg_name}}.data.csv_dataset import CsvDataset, CropType
from {{cookiecutter.pkg_name}}.data.split import Split
from {{cookiecutter.pkg_name}}.model.config import Config, ALL_CONFIGS
from {{cookiecutter.pkg_name}}.model.model import Model
from {{cookiecutter.pkg_name}}.utils.run_once import run_once


def make_dataloader(config: Config, split: Split, example: bool, cpus: int):

    # For printing...
    example_str = "(example) " if example else ""
    print(f"{example_str}Creating dataloader for {split}.")

    # Create dataset.
    dataset = CsvDataset(
        split=split,
        example=example,
        input=config.input,
        get_label=False,
        do_normalization=True,
        crop_type=CropType.CENTER,
        feat_seq_len=config.feat_seq_len,
    )
    dataloader = DataLoader(dataset, shuffle=False, num_workers=cpus-1)

    return dataloader


def _find_lowest_idx(ckpt_paths: List[str]):
    # Find lowest validation loss.
    stems = [os.path.splitext(os.path.basename(str(x)))[0] for x in ckpt_paths]
    parts_per_stem = [x.split("-") for x in stems]
    dict_per_stem = [{p.split("=")[0]: p.split("=")[1]
                      for p in parts if "=" in p} for parts in parts_per_stem]
    val_loss_per_stem = [x["val_loss"] for x in dict_per_stem]
    lowest_idx = min(range(len(val_loss_per_stem)),
                     key=val_loss_per_stem.__getitem__)
    return lowest_idx


def _predict_model(config: Config, example: bool, split: Split, cpus: int):
    example_name = "_example" if example else ""
    example_str = "(example) " if example else ""

    # Device for model computations.
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"{example_str}Using: %s" % device)

    # Load best model.
    # Source: https://github.com/PyTorchLightning/pytorch-lightning/issues/924#issuecomment-591108496
    model_name = f"trained_model_{config.name}{example_name}"
    model_dir = constants.MODELS_DIR.joinpath(model_name)
    ckpt_paths = list(model_dir.glob("best*.ckpt"))
    if len(ckpt_paths) == 0:
        raise Exception(f"No checkpoint path found in {model_dir}.")
    lowest_val_idx = _find_lowest_idx(ckpt_paths)
    ckpt_path = ckpt_paths[lowest_val_idx]
    model = Model.load_from_checkpoint(str(ckpt_path)).to(device)

    # Create dataloader.
    dl = make_dataloader(config, split, example, cpus)

    # Output path.
    dataset = constants.get_dataset(split, example)
    out_file = f"prediction_{config.name}_{split}{example_name}.csv"
    out_path = dataset.predictions_dir.joinpath(out_file)

    # Iterate through data.
    model.eval()
    with open(out_path, mode="w", encoding="utf8") as f:
        f.write("prediction" + "\n")
        for features, _ in tqdm(dl):
            out: torch.Tensor = model(features.to(device)).cpu()
            # TODO: De-normalization operation goes here:
            out_denorm = out
            f.write("%0.7f" % out_denorm.item() + "\n")


def predict_model(config: Config, example: bool, split: Split, cpus: int):

    # Flag name. Make sure this operation is only performed once.
    example_name = "_example" if example else ""
    example_str = "(example) " if example else ""
    flag_name = f"predicted_model_{config.name}_{split}{example_name}"

    # Run exactly once.
    with run_once(flag_name) as should_run:
        if should_run:
            _predict_model(config, example, split, cpus)
        else:
            print(
                f"{example_str}Prediction already made for {config.name} on split {split}.")


if __name__ == "__main__":
    example: bool = True
    cpus: int = 4
    for config in ALL_CONFIGS:
        for split in [Split.VAL, Split.TEST]:
            predict_model(config, example, split, cpus)
