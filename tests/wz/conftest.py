# pylint: disable=redefined-outer-name
import pandas as pd
import pytest

from publicplan.paths import MOCK_DATA_DIR
from publicplan.wz.description import WZCode, build_descriptions
from publicplan.wz.editing import WZAddition
from publicplan.wz.data_processing import ihk_keywords
from publicplan.wz.description import build_gp2019a

from ..conftest import MockClf


@pytest.fixture
def wz_descs():
    return build_descriptions(
        descs_path=MOCK_DATA_DIR.joinpath("wz_desc.xml"),
        keywords_path=MOCK_DATA_DIR.joinpath("wz_keywords.xml"))


@pytest.fixture
def accepted_additions():
    return {
        WZCode("01.12.0"):
        WZAddition(code="01.12.0",
                   explanation="test",
                   keywords=["added keyword"])
    }


@pytest.fixture
def rejected_additions():
    return [{"code": "abc", "explanation": "test", "keywords": ["test"]}]


@pytest.fixture
def codes(wz_descs):
    return wz_descs.codes


@pytest.fixture
def wz_mock_clf(codes):
    return MockClf(codes)


@pytest.fixture
def ihk_descs():
    return pd.read_csv(MOCK_DATA_DIR.joinpath("wz_ihk.csv"),
                       dtype={"Schlüssel": "object"})


@pytest.fixture
def ihk_kws(ihk_descs):
    return ihk_keywords(ihk_descs)


@pytest.fixture
def gp2019a_fixture():
    return build_gp2019a(MOCK_DATA_DIR.joinpath("gp2019a.xml"))
