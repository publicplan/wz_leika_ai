import json
import logging
from pathlib import Path
from typing import List, Optional, Union

import click
import pandas as pd
import torch

from publicplan.models.bert import BertPool
from publicplan.models.config import ModelConfig, TrainConfig
from publicplan.models.dataset import QueryData
from publicplan.models.embeddings import StringCache
from publicplan.models.metrics import mean_avg_prec, mean_recall_at
from publicplan.models.train import test_clf, train_clf
from publicplan.paths import (WZ_TRAIN_PATH, WZ_VAL_PATH, WZ_WEIGHTS_DIR,
                              WZ_GP2019A_PATH, checkpoint_dir)

from .description import ANG_CODES, WZCode, build_descriptions, build_gp2019a
from .data_processing import ihk_keywords

logger = logging.getLogger(__name__)


def load_dataset(csv_path: Union[Path, str],
                 sample_size: int = 0,
                 no_ang: bool = False) -> List[QueryData]:
    """Load dataset from path.

    Args:
    sample_size: Reduce dataset to this size by random sampling.
    no_ang: Exclude data with a. n. g. labels.

    Returns:
        Parsed Dataset
    """
    data = pd.read_csv(csv_path, index_col=0)[["Taetigkeit", "FullCode"]]
    if no_ang:
        data = data[data["FullCode"].apply(lambda x: x not in ANG_CODES)]
    if 0 < sample_size < len(data):
        data = data.sample(n=sample_size, random_state=42)

    return [
        QueryData(query, {int(WZCode(code))})
        for query, code in data.itertuples(index=False)
    ]


@click.command()
@click.option("--config-from",
              "-c",
              "config_dir",
              default=None,
              type=click.Path(exists=True),
              help="Load configuration from directory")
@click.option("--workers", "-w", default=0, type=int, help="Number of workers")
@click.option("--save", "-s", is_flag=True, help="Save model after training")
@click.option("--data-size",
              "-n",
              type=int,
              default=0,
              help="Size of training data set")
@click.option("--no-ang",
              is_flag=True,
              help="Remove ang codes from training data")
def cli(config_dir: Optional[str], save: bool, workers: int, data_size: int,
        no_ang: bool) -> None:
    """Train WZ model with given hyperparameter configuration."""

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    logger.info("Loading data.")

    train_data = load_dataset(WZ_TRAIN_PATH,
                              sample_size=data_size,
                              no_ang=no_ang)
    val_data = load_dataset(WZ_VAL_PATH, sample_size=data_size // 5)

    descs = build_descriptions()
    for code, kws in ihk_keywords().items():
        descs[code].keywords += kws

    for code, kws in build_gp2019a(keywords_path=WZ_GP2019A_PATH).items():
        descs[code].keywords += kws

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
        weights_dir = checkpoint_dir(WZ_WEIGHTS_DIR)

    logger.info("Computing metrics.")

    clf = model_config.build_model(descs)
    trained_clf = train_clf(clf,
                            train_data=train_data,
                            train_config=train_config,
                            val_data=val_data,
                            weights_dir=weights_dir,
                            workers=workers)

    scores = test_clf(trained_clf, val_data, metrics)
    logger.info("Metrics")
    for desc in scores.keys():
        logger.info(desc + f": {scores[desc]:.3f}")

    logger.info("Weights")
    for desc, weight in trained_clf.describe_weights().items():
        logger.info(desc + f": {weight:.3f}")

    if weights_dir is not None:
        logger.info("Saving weights and model config")
        model_config.save(weights_dir)
        train_config.save(weights_dir)

        trained_clf.save_checkpoint(weights_dir)

        description = trained_clf.describe_weights()
        description = {**scores, **description}
        description["no_ang"] = no_ang
        description["train_size"] = len(train_data)
        description["val_size"] = len(val_data)
        json.dump(description,
                  open(weights_dir.joinpath("description.json"), "w"),
                  indent=4)


@click.command()
@click.option("--model_checkpoint",
              "-c",
              type=click.Path(exists=True),
              help="Model checkpoint")
def save_bert_cache(model_checkpoint: str):
    """Save embedding cache for finetuned BERT model."""

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    checkpoint = Path(model_checkpoint)

    model_config = ModelConfig.from_checkpoint(checkpoint)
    if model_config.model_name != "BertFinetuning":
        raise ValueError(
            "Caching of embedding only available for BertFinetuning.")
    pool = BertPool(checkpoint, pooling=model_config.params["pooling"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pool.to(device)
    descs = build_descriptions()
    for code, kws in ihk_keywords().items():
        descs[code].keywords += kws

    for code, kws in build_gp2019a(keywords_path=WZ_GP2019A_PATH).items():
        descs[code].keywords += kws

    if checkpoint.exists() and checkpoint.joinpath("cache.pt").exists():
        cache = StringCache.from_path(checkpoint, device=device)
    else:
        cache = StringCache()
    cache.update_collection(descs, embedding=pool)
    cache.save(checkpoint)
