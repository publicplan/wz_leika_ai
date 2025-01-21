# pylint: disable=unused-argument,redefined-outer-name
import json

import pytest

from publicplan.leika.description import parse_descs
from publicplan.paths import LEIKA_DESCS_PATH

from .conftest import get_response

pytestmark = [pytest.mark.integration, pytest.mark.leika]

ELASTIC_HOST = "elasticsearch"
ELASTIC_PORT = "9200"


@pytest.fixture
def parsed_descs():
    raw_descs = json.load(open(LEIKA_DESCS_PATH))
    return parse_descs(raw_descs)


@pytest.fixture
def raw_descs():
    raw_descs = json.load(open(LEIKA_DESCS_PATH))
    return parse_descs(raw_descs)


def test_api_up(api_up):
    assert True


def test_suche(api_up):

    query = "test"
    query += r"\\n\b!\"§4%&/()n=?ßüäö#+~*-.,;:-"
    query += "%20 %2F"
    num_returned = 15
    params = {"query": query, "num_returned": num_returned}
    result = get_response("suche", params=params)
    assert isinstance(result, list)
    assert len(result) == num_returned


def test_codes(api_up, parsed_descs):
    descs = parsed_descs[1].sort(key=lambda desc: int(desc["schluessel"]))
    result = get_response("codes").sort(
        key=lambda desc: int(desc["schluessel"]))
    assert result == descs


def test_rejected(api_up, parsed_descs):
    rejected = parsed_descs[2]
    result = get_response("rejected")
    assert len(rejected) == len(result)
    for r in rejected:
        assert r in result


def test_desc(api_up, parsed_descs):
    descs = parsed_descs[1]

    code = descs[0]["schluessel"]
    result = get_response(f"descs/{code}")
    assert result in descs
