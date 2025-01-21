# pylint: disable=redefined-outer-name
import pytest

from publicplan.models.dataset import get_dataloader


@pytest.fixture
def score_dataloader(dataset, score_features):
    return get_dataloader(dataset,
                          score_features,
                          sample_size=4,
                          queries_per_batch=2)


@pytest.fixture
def bert_dataloader(dataset, bert_features):
    return get_dataloader(dataset,
                          bert_features,
                          sample_size=4,
                          queries_per_batch=2)


def test_score_dataloader(score_dataloader, score_features):

    batch = next(iter(score_dataloader))
    labels, inputs = batch
    assert labels.size() == (8, 1)
    assert ((labels == 0) | (labels == 1)).all()

    d = score_features.dim
    assert inputs.size() == (8, d)


def test_bert_dataloader(bert_dataloader):

    batch = next(iter(bert_dataloader))
    _, query_ids, *doc_ids = batch

    assert query_ids.dim() == 2
    assert query_ids.size(0) == 8

    for field_ids in doc_ids:
        assert field_ids.dim() == 3
        assert field_ids.size(0) == 8
