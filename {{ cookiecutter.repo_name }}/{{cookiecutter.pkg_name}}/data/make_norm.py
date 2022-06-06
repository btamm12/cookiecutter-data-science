# -*- coding: utf-8 -*-
import click
from dotenv import find_dotenv, load_dotenv
import logging
import os

from {{cookiecutter.pkg_name}}.data.norm.calculate_norm import calculate_norm
from {{cookiecutter.pkg_name}}.data.split import Split


@click.command()
@click.option('-e', '--example', is_flag=True)
def main(example):
    """Calculate norm and variance of each channel over all inputs in the training
    split."""
    logger = logging.getLogger(__name__)
    logger.info(f"Running {os.path.basename(__file__)}")
    logger.info("Calculating norm and variance for each split.")

    # Calculate norm.
    calculate_norm(Split.TRAIN, example)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
