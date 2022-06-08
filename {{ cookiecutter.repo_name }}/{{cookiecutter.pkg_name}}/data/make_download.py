# -*- coding: utf-8 -*-
import click
from dotenv import find_dotenv, load_dotenv
import logging
import os


@click.command()
def main():
    """Download dataset(s) and save the results in the following folder:

    ```
    {{cookiecutter.repo_name}}/data/raw/
    ```
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Running {os.path.basename(__file__)}")
    logger.info("Downloading dataset(s).")

    # TODO: call download scripts here
    logger.info("[No download commands]")
    pass


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
