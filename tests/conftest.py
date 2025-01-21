# pylint: disable=redefined-outer-name
import json

import numpy as np
import pandas as pd
import pytest

from publicplan.leika.description import parse_descs
from publicplan.leika.train import load_dataset
from publicplan.paths import MOCK_DATA_DIR


@pytest.fixture
def raw_descriptions():
    desc_path = MOCK_DATA_DIR.joinpath("leika_desc.json")
    return json.load(open(desc_path))


@pytest.fixture
def descriptions(raw_descriptions):
    return parse_descs(raw_descriptions)[0]


@pytest.fixture
def descriptions2(raw_descriptions):
    # raw_descriptions...
    return parse_descs(raw_descriptions)[0]


@pytest.fixture
def codes():
    return [
        99117035131000,
        99119001035000,
        99097006000000,
        99097006047000,
        99097006080000,
    ]


@pytest.fixture
def dataset():

    data_path = MOCK_DATA_DIR.joinpath("leika_data.json")
    return load_dataset(data_path)


class MockSpellCheck:

    # pylint: disable=no-self-use
    def correct(self, s):
        return s


@pytest.fixture
def mock_spellcheck():
    return MockSpellCheck()


class MockClf:
    def __init__(self, codes):
        self.codes = codes

    # pylint: disable=unused-argument
    def predict(self, query, num_returned=None):
        if num_returned is None:
            num_returned = len(self.codes)

        scores = np.arange(0, 1, 1 / len(self.codes))[::-1][:num_returned]
        codes = self.codes[:num_returned]

        return pd.Series(scores, index=codes)

    # pylint: disable=unused-argument
    def batch_predict(self, queries, num_returned=None):
        return [self.predict("", num_returned=num_returned)] * len(queries)

    def to_device(self, device):
        pass


@pytest.fixture
def mock_clf(codes):
    return MockClf(codes)
