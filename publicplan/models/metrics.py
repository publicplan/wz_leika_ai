from typing import Callable, List, Tuple

import torch
from tqdm import tqdm

from .classifiers import Classifier
from .dataset import QueryData

RankedResults = List[Tuple[List[int], List[int]]]
Metric = Callable[[RankedResults], float]


def compute_predictions(clf: Classifier,
                        dataset: List[QueryData],
                        batch_size: int = 10,
                        num_preds: int = 0,
                        pbar: bool = True) -> RankedResults:
    """Compute predictions of classifier.

    Args:
        clf: Classifier to use.
        dataset: Dataset of annotated queries.
        batch_size: Size of batch predicted simultaneously.
        num_preds: If non-null, only return this many predictions
        pbar: Whether to display a progress bar.

    Returns:
        List of pairs of labelled codes and ranked predictions.
    """

    # clf.to_device(torch.device("cpu"))
    results: RankedResults = []
    num_batches = len(dataset) // batch_size

    for n in tqdm(range(num_batches),
                  ncols=100,
                  dynamic_ncols=True,
                  disable=not pbar):
        batch = dataset[n * batch_size:(n + 1) * batch_size]
        preds = clf.batch_predict([qd.query for qd in batch])
        if num_preds:
            preds = preds[:num_preds]
        result = [(list(qd.relevant), pred.index.tolist())
                  for qd, pred in zip(batch, preds)]
        results += result
    rest = dataset[num_batches * batch_size:]
    if rest:
        preds = clf.batch_predict([qd.query for qd in rest])
        result = [(list(qd.relevant), pred.index.tolist())
                  for qd, pred in zip(rest, preds)]
        results += result

    return results


def mean_avg_prec(n: int = 10) -> Metric:
    """Build mean average precision metric.

    See https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
    for a definition.

    Args:
        n: Only consider codes up to this rank.

    Returns:
        Function taking a list of relevant codes and ranked  predictions
        and returning the corresponding mean average precision metric.
    """
    def acc_metric(relevant: List[int], pred_codes: List[int],
                   k: int) -> float:
        if pred_codes[k - 1] in relevant:
            num_predicted = len(
                [code for code in pred_codes[:k] if code in relevant])
            prec_at_k = num_predicted / len(relevant)
            return prec_at_k / k
        return 0

    def metric(results: RankedResults) -> float:
        return _mean_metric(results, 1, n, acc_metric)

    return metric


def mean_recall_at(n) -> Metric:
    """Compute mean recall among codes with rank <= n.

    The recall for a single query is defined as:

    recall = #relevant_codes_in_top_n / #total_relevant_codes


    Args:
        n: Only consider codes up to this rank.

    Returns:
        Function taking a list of relevant codes and ranked predictions
        and returning the corresponding mean recall metric.
    """
    def acc_metric(relevant: List[int], pred_codes: List[int],
                   _k: int) -> float:
        num_relevant_preds = len(
            [code for code in pred_codes[:n] if code in relevant])
        return num_relevant_preds / len(relevant)

    def metric(results: RankedResults) -> float:
        return _mean_metric(results, n, n, acc_metric)

    return metric


_AccMetric = Callable[[List[int], List[int], int], float]


def _mean_metric(results: RankedResults, min_k: int, max_k: int,
                 metric: _AccMetric) -> float:

    summed: float = 0
    for relevant, pred_codes in results:
        for k in range(min_k, max_k + 1):
            summed += metric(relevant, pred_codes, k)

    return summed / len(results)


def precision(preds: torch.Tensor,
              labels: torch.Tensor,
              eps: float = 1.e-5) -> float:
    """Binary precision for batch of model predictions.

    Binary Precision is given by:

    precision = #true_positives/(#true_positives + #false_positives)
    """

    tp, _, fp, _ = _binary_comparison(preds, labels)
    return tp / (tp + fp + eps)


def recall(preds: torch.Tensor,
           labels: torch.Tensor,
           eps: float = 1.e-5) -> float:
    """Binary recall for batch of model predictions.

    Binary recall is given by:

    recall = #true_positives/(#true_positives + #false_negatives)
    """

    tp, _, _, fn = _binary_comparison(preds, labels)
    return tp / (tp + fn + eps)


def _binary_comparison(preds: torch.Tensor,
                       labels: torch.Tensor) -> Tuple[int, int, int, int]:
    preds = preds > 0.5
    labels = labels > 0.5
    tp = (preds & labels).sum().item()
    tn = (~preds & ~labels).sum().item()
    fp = (preds & ~labels).sum().item()
    fn = (~preds & labels).sum().item()

    return tp, tn, fp, fn  #type: ignore
