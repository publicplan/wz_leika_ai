# pylint: disable=unused-argument,redefined-outer-name
import json
from pathlib import Path

import pytest
import requests

from publicplan.wz.api import LogAddition
from publicplan.wz.description import build_descriptions
from publicplan.wz.editing import WZAddition

from .conftest import BASE_URL, get_response
from .serve_editing import wz_additions

pytestmark = [pytest.mark.integration, pytest.mark.wz]

LOG_DIR = "/api_log"


def compare_descriptions(desc1, desc2):
    # Descriptions might differ in the keyword field due to the update
    return all(desc1[f] == desc2[f] for f in desc1.keys() if f != "keywords")


@pytest.fixture
def descs():
    return build_descriptions()


@pytest.fixture
def editing_correct():
    return wz_additions()[:2]


@pytest.fixture
def editing_malformed():
    return wz_additions()[2]


def test_api_up(api_up):
    assert True


def test_suche(api_up):

    query = "test"
    query += r"\\n\b!\"§4%&/()n=?ßüäö#+~*-.,;:-"
    query += "%20 %2F"
    num_returned = 15
    params = {"query": query, "num_returned": num_returned}
    response = get_response("suche", params=params)
    results = response["results"]
    assert isinstance(results, list)
    assert len(results) == num_returned


@pytest.mark.skipif(not Path(LOG_DIR).exists(),
                    reason="Test needs docker volume setup.")
def test_log(api_up):
    query = "test"
    user_token = "abc"
    source = "test_source"
    params = {"query": query, "user_token": user_token, "source": source}
    response = get_response("suche", params=params)
    query_id = response["query_id"]
    log_path = Path(LOG_DIR).joinpath(source, query_id + ".json")
    assert log_path.exists()

    log = json.load(open(log_path, "r"))

    assert query == log["query"]
    assert response["results"] == log["results"]


@pytest.mark.skipif(not Path(LOG_DIR).exists(),
                    reason="Test needs docker volume setup.")
def test_log_add(api_up):
    query = "test"
    user_token = "abc"
    source = "test_source"
    params = {"query": query, "user_token": user_token, "source": source}
    response = get_response("suche", params=params)
    query_id = response["query_id"]
    log_path = Path(LOG_DIR).joinpath(source, query_id + ".json")

    original_log = json.load(open(log_path, "r"))
    codes = [
        result["occupation"]["code"] for result in original_log["results"]
    ]
    for user_action in codes + [LogAddition.no_match]:
        log_body = {
            "query_id": query_id,
            "source": source,
            "user_action": user_action
        }
        log_response = requests.post(f"{BASE_URL}/log_add/", json=log_body)

        new_log = json.load(open(log_path, "r"))
        assert log_response.status_code == 200
        assert new_log == {**original_log, **{"user_action": user_action}}


def test_all(api_up, descs):
    result = get_response("descs/")
    assert len(result) == len(descs.codes)
    assert all(
        compare_descriptions(desc1, descs[code].dict())
        for desc1, code in zip(result, descs.codes))


def test_descs(api_up, descs):
    codes = [desc.code for desc in list(descs.values())[:10]]
    for code in codes:
        result = get_response(f"descs/{code}")
        assert compare_descriptions(result, descs[code].dict())


def test_editing_accepted(api_up, editing_correct):
    result = get_response("editing/accepted")
    added = [WZAddition(**add) for add in editing_correct]
    expected = {add.code: add.dict() for add in added}
    assert result == expected


def test_editing_rejected(api_up, editing_malformed):
    result = get_response("editing/rejected")
    assert result == [editing_malformed]


def test_editing(api_up, editing_correct):
    query = "Test"
    params = {"query": query, "num_returned": 5}
    response = get_response("suche", params=params)
    results = response["results"]
    assert results[0]["occupation"]["code"] == editing_correct[0]["code"]
