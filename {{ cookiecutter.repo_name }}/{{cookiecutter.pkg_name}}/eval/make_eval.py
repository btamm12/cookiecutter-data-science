# -*- coding: utf-8 -*-
import click
from dotenv import find_dotenv, load_dotenv
import logging
import os

from {{cookiecutter.pkg_name}}.eval.combine_csvs import combine_csvs
from {{cookiecutter.pkg_name}}.eval.eval import eval


@click.command()
def main():
    """Evaluate models on the validation set(s) and print the results to STDOUT."""
    logger = logging.getLogger(__name__)
    logger.info(f"Running {os.path.basename(__file__)}")
    logger.info('Evaluating models on validation set(s).')

    # Combine ground-truth + prediction CSVs.
    combine_csvs()

    # Evaluate combined CSV's.
    eval()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
