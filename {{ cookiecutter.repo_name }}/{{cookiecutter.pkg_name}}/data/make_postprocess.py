# -*- coding: utf-8 -*-
import click
from dotenv import find_dotenv, load_dotenv
import logging
import os


@click.command()
def main():
    """Perform postprocessing on the model predictions to format them correctly for
    submission. The results will be saved in the following folder:

    ```
    {{cookiecutter.repo_name}}/data/submission/
    ```
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Running {os.path.basename(__file__)}")
    logger.info("Performing postprocessing on the model predictions.")

    # TODO: call postprocessing scripts here
    pass


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
