# -*- coding: utf-8 -*-
import click
from dotenv import find_dotenv, load_dotenv
import logging
import os

from {{cookiecutter.pkg_name}}.model.config import ALL_CONFIGS
from {{cookiecutter.pkg_name}}.train.train_model import train_model


@click.command()
@click.option('-e', '--example', is_flag=True)
@click.option('-i', '--partition_idx', default=0)
@click.option('-n', '--num_partitions', default=1)
@click.option('-c', '--cpus', default=4)
def main(example, partition_idx, num_partitions, cpus):
    """Train models and save the results in the following folder:
    
    ```
    {{cookiecutter.repo_name}}/models/
    ```

    The logs will be stored in

    ```
    {{cookiecutter.repo_name}}/lightning_logs/
    ```
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Running {os.path.basename(__file__)}")
    logger.info('Training models.')

    # Do training for each config.
    # Determine which jobs are for this partition.
    jobs = ALL_CONFIGS    
    N = len(jobs)
    start_idx = int(partition_idx*N/num_partitions)
    end_idx = int((partition_idx+1)*N/num_partitions)
    jobs_i = jobs[start_idx:end_idx]

    # Train models.
    for job in jobs_i:
        config = job
        print(f"Training config: {config.name}.")
        train_model(config, example, cpus)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
