# pylint: disable=redefined-outer-name
import pytest

from publicplan.models.dataset import QueryData
from publicplan.models.metrics import (compute_predictions, mean_avg_prec,
                                       mean_recall_at)


@pytest.fixture
def dataset(codes):
    return [QueryData("test", set([codes[0], codes[2]]))]


@pytest.fixture
def results(mock_clf, dataset):
    return compute_predictions(mock_clf, dataset, pbar=False)


def test_mean_avg_prec(results):

    m = mean_avg_prec(3)(results)

    assert m == (1 / 2) * (1 + 1 / 3 * 2)


def test_mean_recall_at(results):

    m = mean_recall_at(3)(results)
    assert m == 1

    m = mean_recall_at(2)(results)
    assert m == 1 / 2
