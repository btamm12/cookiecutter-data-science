# -*- coding: utf-8 -*-
import click
from dotenv import find_dotenv, load_dotenv
import logging
import os


@click.command()
@click.option("-e", "--example", is_flag=True)
@click.option("-i", "--partition_idx", default=0)
@click.option("-n", "--num_partitions", default=1)
def main(example, partition_idx, num_partitions):
    """Perform feature extraction on the preprocessed data. These features will be
    the input for the training. The features will be saved in the following folder:

    ```
    {{cookiecutter.repo_name}}/data/processed/
    ```
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Running {os.path.basename(__file__)}")
    logger.info("Performing feature extraction on preprocessed data.")

    # Extract features.
    # TODO: call feature extraction code here
    pass

    # from {{cookiecutter.pkg_name}}.data.split import Split
    # extract_features(Split.TRAIN, example, partition_idx, num_partitions)
    # extract_features(Split.VAL, example, partition_idx, num_partitions)
    # extract_features(Split.TEST, example, partition_idx, num_partitions)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
