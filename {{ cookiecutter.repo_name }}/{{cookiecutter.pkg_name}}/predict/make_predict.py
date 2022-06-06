# -*- coding: utf-8 -*-
import click
from dotenv import find_dotenv, load_dotenv
import logging
import os

from {{cookiecutter.pkg_name}}.model.config import ALL_CONFIGS
from {{cookiecutter.pkg_name}}.predict.predict_model import predict_model
from {{cookiecutter.pkg_name}}.utils.split import Split


@click.command()
@click.option("-e", "--example", is_flag=True)
@click.option("-i", "--partition_idx", default=0)
@click.option("-n", "--num_partitions", default=1)
@click.option("-c", "--cpus", default=4)
def main(example, partition_idx, num_partitions, cpus):
    """Make model predictions on validation and test splits."""
    logger = logging.getLogger(__name__)
    logger.info(f"Running {os.path.basename(__file__)}")
    logger.info("Making model predictions on validation/test splits.")

    # Do prediction for each config on the validation/test splits.
    # Determine which jobs are for this partition.
    jobs = [
        (config, split) for config in ALL_CONFIGS for split in [Split.VAL, Split.TEST]
    ]
    N = len(jobs)
    start_idx = int(partition_idx * N / num_partitions)
    end_idx = int((partition_idx + 1) * N / num_partitions)
    jobs_i = jobs[start_idx:end_idx]

    # Train models.
    for job in jobs_i:
        config, split = job
        print(f"Predicting config: {config.name} on split {split}.")
        predict_model(config, example, split, cpus)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
