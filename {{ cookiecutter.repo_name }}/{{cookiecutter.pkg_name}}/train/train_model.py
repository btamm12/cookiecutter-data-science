import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from torch.utils.data import DataLoader

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
        get_label=True,
        integer_label=config.label.is_integer,
        do_normalization=True,
        crop_type=CropType.CENTER,
        feat_seq_len=config.feat_seq_len,
    )
    dataloader = DataLoader(dataset, shuffle=False, num_workers=cpus-1)

    return dataloader


def _train_model(config: Config, example: bool, cpus: int):

    # Create model.
    model = Model(config)

    # Create dataloader(s).
    train_dl = make_dataloader(config, Split.TRAIN, example, cpus)
    val_dl = make_dataloader(config, Split.VAL, example, cpus)

    # Trainer parameters.
    example_name = "_example" if example else ""
    example_str = "(example) " if example else ""
    out_name = f"trained_model_{config.name}{example_name}"
    model_dir = constants.MODELS_DIR.joinpath(out_name)

    best_ckpt_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=str(model_dir),
        filename="best-{epoch:03d}-{val_loss:.6f}",
        save_top_k=3,
        mode="min",
    )
    last_ckpt_callback = ModelCheckpoint(
        dirpath=str(model_dir),
        filename="last",
    )
    # Device for model computations.
    if torch.cuda.is_available():
        gpus = 1
        device = "cuda"
    else:
        gpus = 0
        device = "cpu"
    print(f"{example_str}Using: %s" % device)

    trainer_params = {
        "gpus": gpus,
        "max_epochs": config.train_config.max_epochs,
        "weights_save_path": str(),
        # "strategy": "ddp",  # distributed computing
        "callbacks": [best_ckpt_callback, last_ckpt_callback],
        "progress_bar_refresh_rate": 50,
        "weights_summary": "full",
    }
    trainer = pl.Trainer(**trainer_params)

    last_ckpt_path = model_dir.joinpath("last.ckpt")
    if last_ckpt_path.exists():
        trainer.fit(model, train_dl, val_dl, ckpt_path=str(last_ckpt_path))
    else:
        trainer.fit(model, train_dl, val_dl)


def train_model(config: Config, example: bool, cpus: int):

    # Flag name. Make sure this operation is only performed once.
    example_name = "_example" if example else ""
    example_str = "(example) " if example else ""
    flag_name = f"trained_model_{config.name}{example_name}"

    # Run exactly once.
    with run_once(flag_name) as should_run:
        if should_run:
            _train_model(config, example, cpus)
        else:
            print(f"{example_str}Model already trained for {config.name}.")


if __name__ == "__main__":
    example: bool = True
    cpus: int = 4
    for config in ALL_CONFIGS:
        train_model(config, example, cpus)
