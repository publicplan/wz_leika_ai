import json
import logging
from pathlib import Path
from typing import List, Optional, Union

import click

from publicplan.models.config import ModelConfig, TrainConfig
from publicplan.models.dataset import QueryData
from publicplan.models.metrics import mean_avg_prec, mean_recall_at
from publicplan.models.train import save_clf, test_clf, train_clf

from ..paths import (LEIKA_CLEANED_DESCS_PATH, LEIKA_TRAIN_PATH,
                     LEIKA_VAL_PATH, LEIKA_WEIGHTS_DIR, checkpoint_dir)
from .description import LeikaDescriptions

logger = logging.getLogger(__name__)


def load_dataset(path: Union[Path, str]) -> List[QueryData]:
    """Load dataset from json file."""
    data_json = json.load(open(path, "r"))

    return [QueryData(d['query'], set(d['codes'])) for d in data_json]


@click.command()
@click.option("--config-from",
              "-c",
              "config_dir",
              default=None,
              type=click.Path(exists=True),
              help="Load configuration from directory")
@click.option("--workers", "-w", default=0, type=int, help="Number of workers")
@click.option("--save", "-s", is_flag=True, help="Save model after training")
def cli(config_dir: Optional[str], save: bool, workers: int) -> None:
    """Train LeiKa model with given hyperparameter configuration."""

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    logger.info("Loading data.")

    train_data = load_dataset(LEIKA_TRAIN_PATH)
    val_data = load_dataset(LEIKA_VAL_PATH)

    descs = LeikaDescriptions.from_path(LEIKA_CLEANED_DESCS_PATH)
    train_config = TrainConfig.from_dir(config_dir)
    model_config = ModelConfig.from_dir(config_dir)

    metrics = {
        "Mean recall at 1": mean_recall_at(1),
        "Mean recall at 5": mean_recall_at(5),
        "Mean recall at 100": mean_recall_at(100),
        "Mean average precision": mean_avg_prec(10)
    }

    weights_dir = None
    if save:
        weights_dir = checkpoint_dir(LEIKA_WEIGHTS_DIR)

    clf = model_config.build_model(descs)
    trained_clf = train_clf(clf,
                            train_data=train_data,
                            train_config=train_config,
                            workers=workers)

    scores = test_clf(trained_clf, val_data, metrics)
    logger.info("Metrics")
    for desc in scores.keys():
        logger.info(desc + f": {scores[desc]:.3f}")

    logger.info("Weights")
    for desc, weight in trained_clf.describe_weights().items():
        logger.info(desc + f": {weight:.3f}")

    if weights_dir is not None:
        checkpoint = checkpoint_dir(weights_dir)
        logger.info("Saving weights and model config")
        model_config.save(checkpoint)
        train_config.save(checkpoint)
        save_clf(trained_clf, checkpoint, scores)
