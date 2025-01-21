import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer  # type: ignore
from tqdm import tqdm

from publicplan.paths import checkpoint_dir

from .classifiers import Classifier
from .config import TrainConfig
from .dataset import QueryData, get_dataloader
from .metrics import (Metric, compute_predictions, mean_avg_prec,
                      mean_recall_at, precision, recall)

logger = logging.getLogger(__name__)


def default_metrics() -> Dict[str, Metric]:
    """Default metrics to use"""
    return {
        "Mean recall at 1": mean_recall_at(1),
        "Mean recall at 5": mean_recall_at(5),
        "Mean recall at 100": mean_recall_at(100),
        "Mean average precision": mean_avg_prec(10)
    }


# pylint: disable=too-many-locals
def train_clf(clf: Classifier,
              train_data: List[QueryData],
              train_config: TrainConfig,
              val_data: Optional[List[QueryData]] = None,
              val_step: int = 10000,
              weights_dir: Optional[Path] = None,
              workers: int = 0) -> Classifier:
    """Train model and compute metrics on validation set.

    Args:
        clf: Model to train.
        train_data: Training data to use.
        train_config: Training hyperparameters.
        val_data: Validation data to use.
        val_step: Save model and compute metrics after this many steps.
        weights_dir: If not None, save model checkpoint in this directory,
            after every validation step.
        workers: Number of workers for data loading

    Returns:
        Trained classifier.
    """

    criterion = nn.BCELoss()
    optimizer = clf.optimizer(train_config.lr, train_config.weight_decay)
    train_dl = get_dataloader(train_data,
                              clf.features,
                              train_config.sample_size,
                              train_config.queries_per_batch,
                              num_workers=workers)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    clf.to(device)

    training_scores = {"Loss": 0.0, "Precision": 0.0, "Recall": 0.0}
    total = len(train_data) // train_config.queries_per_batch
    val_step = val_step // train_config.queries_per_batch

    checkpoints_dir: Optional[Path] = None
    writer = None
    if weights_dir:
        run_dir = weights_dir.joinpath("runs")
        run_dir.mkdir(exist_ok=True)
        try:
            # pylint: disable=import-outside-toplevel
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(run_dir)
        except ImportError:
            logger.warning("Tensorboard not installed.")
        checkpoints_dir = weights_dir.joinpath("checkpoints")
        checkpoints_dir.mkdir(exist_ok=True)

    for e in range(1, train_config.epochs + 1):

        logger.info(f"Starting epoch {e}/{train_config.epochs}")

        progress_bar = tqdm(enumerate(train_dl),
                            total=total,
                            dynamic_ncols=True)
        for n, batch in progress_bar:
            if val_data and n and n % val_step == 0:
                clf.update_cache()
                scores = test_clf(clf, val_data)
                print("\nMetrics")
                for desc in scores.keys():
                    print(desc + f": {scores[desc]:.3f}")
                if checkpoints_dir is not None:
                    checkpoint = checkpoint_dir(checkpoints_dir)
                    save_clf(clf, checkpoint)
                if writer is not None:
                    writer.add_scalars(main_tag="validation/metrics",
                                       tag_scalar_dict=scores,
                                       global_step=(e - 1) * len(train_dl) + n)
            batch_scores = _training_loop(clf, criterion, optimizer, batch)
            summary = _training_eval(training_scores, batch_scores)
            progress_bar.set_description(summary)
            if writer is not None:
                writer.add_scalars(main_tag="train/scores",
                                   tag_scalar_dict=batch_scores,
                                   global_step=(e - 1) * len(train_dl) + n)

    return clf


def _training_loop(clf: Classifier, criterion: nn.BCELoss,
                   optimizer: Optimizer,
                   batch: Tuple[torch.Tensor, ...]) -> Dict[str, float]:

    clf.train()
    labels, *inputs = (b.to(clf.device) for b in batch)
    out = clf(*inputs)

    optimizer.zero_grad()
    loss = criterion(out, labels)
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        batch_precision = precision(out, labels)
        batch_recall = recall(out, labels)

    batch_scores = {
        "Loss": loss.item(),
        "Precision": batch_precision,
        "Recall": batch_recall
    }

    return batch_scores


def _training_eval(training_scores: Dict[str, float],
                   batch_scores: Dict[str, float],
                   alpha: float = 0.1) -> str:
    """Log exponential running average of training scores."""
    summary = ""
    for name in training_scores.keys():
        score = alpha * batch_scores[name]
        score += (1 - alpha) * training_scores[name]
        summary += f"{name}: {score:.3f} "
        training_scores[name] = score

    return summary


def save_clf(clf: Classifier,
             checkpoint: Union[Path, str],
             scores: Optional[Dict[str, float]] = None) -> None:
    """Save trained classifier with associated scores.

    Args:
        clf: Trained Classifier.
        checkpoint: Target Directory.
        scores: Computed validation scores. Ignored if None.
    """

    clf.save_checkpoint(checkpoint)

    description = clf.describe_weights()
    if scores:
        description = {**scores, **description}
    json.dump(description,
              open(Path(checkpoint).joinpath("description.json"), "w"),
              indent=4)


def test_clf(clf: Classifier,
             data: List[QueryData],
             metrics: Optional[Dict[str, Metric]] = None) -> Dict[str, float]:
    """Test classifier with given metrics.

    Args:
        clf: Classifier to test.
        data: Test dataset.
        metrics: Pairs of names and metric functions. If none,
            use default metrics.

    Returns:
        Pairs of metric description and score.
    """
    results = compute_predictions(clf, data)
    if metrics is None:
        metrics = default_metrics()

    return {desc: metric(results) for desc, metric in metrics.items()}
