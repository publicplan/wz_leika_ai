# pylint: disable=redefined-outer-name,no-self-use
import numpy as np
import pytest

from publicplan.models.bert import BertFeatures, BertTokenizer
from publicplan.models.classifiers import LinearClassifier
from publicplan.models.embeddings import (CachedEmbeddings, EmbSimil,
                                          SumWordsEmbedding)
from publicplan.models.features import ConcatFeatures
from publicplan.models.termdoc import TermDocs


class MockEmbedding:

    name = "Mock embedding"
    vector_dim = 2

    # pylint: disable=no-self-use, unused-argument
    def vectorize(self, text, normalize=True):
        return np.array([len(text), np.log(1 + len(text))], dtype=np.float32)


@pytest.fixture
def mock_embedding():
    return MockEmbedding()


@pytest.fixture
def swe(mock_embedding):
    return SumWordsEmbedding(mock_embedding)


@pytest.fixture
def cached_embs(descriptions, swe):
    return CachedEmbeddings(descriptions, swe, fields=descriptions.fields)


@pytest.fixture
def emb_simil(cached_embs):

    return EmbSimil(cache=cached_embs)


@pytest.fixture
def termdocs(descriptions):

    td = TermDocs(descriptions)

    return td


@pytest.fixture
def score_features(emb_simil, termdocs):
    return ConcatFeatures([emb_simil, termdocs])


@pytest.fixture
def clf(score_features):
    return LinearClassifier(score_features)


class MockTokenizer:

    max_len_single_sentence = 100

    def tokenize(self, s):
        return s.split(" ")

    def convert_tokens_to_ids(self, tokens):
        return [1] * len(tokens)

    def build_inputs_with_special_tokens(self, ids):
        return [2] + ids + [3]


@pytest.fixture
def bert_tokenizer(monkeypatch):
    monkeypatch.setattr(BertTokenizer, "_load_tokenizer",
                        lambda self, model_checkpoint: MockTokenizer())
    return BertTokenizer("MockedBert")


@pytest.fixture
def bert_features(descriptions, bert_tokenizer):
    features = BertFeatures(descriptions, tokenizer=bert_tokenizer)

    return features
