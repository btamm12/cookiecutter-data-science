# -*- coding: utf-8 -*-
import click
from dotenv import find_dotenv, load_dotenv
import logging
import os


@click.command()
def main():
    """Perform preprocessing on the raw dataset(s) and save the results in the
    following folder:

    ```
    {{cookiecutter.repo_name}}/data/processed/
    ```
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Running {os.path.basename(__file__)}")
    logger.info("Performing preprocessing on the raw dataset(s).")

    # TODO: call preprocessing scripts here
    pass


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
