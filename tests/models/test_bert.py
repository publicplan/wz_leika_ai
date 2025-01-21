# pylint: disable=no-self-use,redefined-outer-name,not-callable
import pytest
import torch

from publicplan.models.bert import BertFinetuning


class MockPool:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.dim = 10

    def __call__(self, ids):
        return ids.float().sum(1).unsqueeze(1).expand(-1, self.dim)

    def vectorize(self, texts):
        ids = self.tokenizer.stack_ids(texts)
        return self(ids)


@pytest.fixture
def pool(bert_tokenizer):
    return MockPool(bert_tokenizer)


@pytest.fixture
def clf(bert_features, pool):
    return BertFinetuning(bert_features, pool)


@pytest.fixture
def feature_ids(bert_features, codes):
    return bert_features.get("test test", codes)


def test_features(feature_ids, codes):

    query_ids, *doc_ids = feature_ids

    assert query_ids.size() == (len(codes), 4)
    for field_ids in doc_ids:
        assert field_ids.size(0) == len(codes)
        assert field_ids.dim() == 3


def test_clf(clf, feature_ids, codes):

    with torch.no_grad():
        results = clf(*feature_ids)

    assert (results >= 0).all()
    assert (results <= 1).all()
    assert results.size() == (len(codes), 1)
