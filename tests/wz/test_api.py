# pylint: disable=redefined-outer-name,undefined-variable

import json
from urllib.parse import quote_plus

import pytest
from starlette.testclient import TestClient

from publicplan.wz.api import LogAddition, SearchResponse, build_wz_api


def format_desc(desc):
    return {
        "code": desc.code,
        "name": desc.name,
        "section_name": desc.section_name,
        "group_name": desc.group_name,
        "class_name": desc.class_name,
        "explanation": desc.explanation,
        "exclusions": desc.exclusions,
        "keywords": desc.keywords
    }


@pytest.fixture(scope="session")
def log_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("wz_api_logs_test")


@pytest.fixture
def app(wz_mock_clf, wz_descs, log_dir, accepted_additions, rejected_additions,
        ihk_kws, gp2019a_fixture):
    app = build_wz_api(wz_mock_clf,
                       wz_descs,
                       ihk_kws=ihk_kws,
                       gp2019a=gp2019a_fixture,
                       accepted_additions=accepted_additions,
                       rejected_additions=rejected_additions,
                       log_dir=log_dir)

    @app.get("/parse/")
    # pylint: disable=unused-variable
    def parse(query: str):
        return query

    return app


@pytest.fixture
def client(app):
    return TestClient(app)


def test_api_is_up(client):
    response = client.get("/")
    assert response.status_code == 200


def test_parse(client):
    query = "Frisörsalon"
    query += r"\\n\b!\"§4%&/()n=?ßüäö#+~*-.,;:-"
    query += "%20 %2F"
    encoded = quote_plus(query)
    response = client.get(f"/parse/?query={encoded}")
    assert response.status_code == 200

    assert response.json() == query


def test_api_query(client):
    query = "Frisörsalon"
    query += r"\\n\b!\"§4%&/()n=?ßüäö#+~*-.,;:-"
    query += "%20 %2F"
    encoded = quote_plus(query)
    num_results = 4
    response = client.get(
        f"/suche/?query={encoded}&num_returned={num_results}")

    assert response.status_code == 200

    response_data = response.json()
    assert SearchResponse(**response_data)

    results = response_data["results"]
    assert len(results) == num_results

    # rounding error tolerance
    eps = 0.001

    for result in results:
        p = result['relevance_confidence']
        assert p >= -eps
        assert p <= 1 + eps

    assert results == sorted(results,
                             key=lambda r: r["relevance_confidence"],
                             reverse=True)


def test_api_query_ihk(client, ihk_kws):
    response = client.get(f"/suche/?query=test&num_returned=4")
    results = response.json()["results"]
    assert any(r["ihk_keywords"] for r in results)
    for r in results:
        assert r["ihk_keywords"] == ihk_kws[r["occupation"]["code"]]


def test_api_query_gp2019a(client, gp2019a_fixture):
    response = client.get(f"/suche/?query=test&num_returned=4")
    results = response.json()["results"]
    print(repr(results))
    assert any(r["gp2019a"] for r in results)
    for r in results:
        assert r["gp2019a"] == gp2019a_fixture[r["occupation"]["code"]]


def test_logging(client, log_dir):
    query = "test"
    source = "test_source"
    user_token = "abc/123"
    response = client.get(
        f"/suche/?query={quote_plus(query)}&source={source}&user_token={user_token}"
    )
    response_data = response.json()
    query_id = response_data["query_id"]
    results = response_data["results"]

    log_path = log_dir.joinpath(source, f"{query_id}.json")

    assert log_path.exists()

    log = json.load(open(log_path, "r"))

    assert query == log["query"]
    assert source == log["source"]
    assert results == log["results"]


def _log_body(query_id, source, user_action):
    return {"query_id": query_id, "source": source, "user_action": user_action}


def test_log_add(client, log_dir):
    query = "test"
    source = "test_source"
    user_token = "abc/123"
    query_response = client.get(
        f"/suche/?query={quote_plus(query)}&source={source}&user_token={user_token}"
    )
    query_id = query_response.json()["query_id"]

    log_path = log_dir.joinpath(source, f"{query_id}.json")
    original_log = json.load(open(log_path, "r"))

    codes = [
        result["occupation"]["code"] for result in original_log["results"]
    ]
    for user_action in codes + [LogAddition.no_match]:
        log_response = client.post("/log_add/",
                                   json=_log_body(query_id, source,
                                                  user_action))

        assert log_response.status_code == 200
        new_log = json.load(open(log_path, "r"))
        assert new_log == {**original_log, **{"user_action": user_action}}

    user_action = "12.12.12"
    log_response = client.post("/log_add/",
                               json=_log_body(query_id, source, user_action))
    assert log_response.status_code == 422
    log_response = client.post(f"/log_add/",
                               json=_log_body("1", source, user_action))
    assert log_response.status_code == 404

    user_action = "abc"
    log_response = client.post("/log_add/",
                               json=_log_body(query_id, source, user_action))
    assert log_response.status_code == 422


def test_all_descs(client, wz_descs):
    response = client.get("/descs/")

    result = response.json()
    expected = [format_desc(wz_descs[code]) for code in wz_descs]
    assert response.status_code == 200
    assert result == expected


def test_code(client, wz_descs):
    code = "01.12.0"
    response = client.get(f"/descs/{code}")

    result = response.json()
    expected = format_desc(wz_descs[code])
    assert response.status_code == 200
    assert result == expected


def test_code_malformed(client):
    for code in ["1100000", "abc"]:
        response = client.get(f"/descs/{code}")
        assert response.status_code == 422


def test_editing_accepted(client, accepted_additions):
    response = client.get("/editing/accepted")
    result = response.json()
    assert response.status_code == 200
    assert result == accepted_additions


def test_editing_rejected(client, rejected_additions):
    response = client.get("/editing/rejected")
    result = response.json()
    assert response.status_code == 200
    assert result == rejected_additions


def test_editing_search_result(client, accepted_additions):
    response = client.get(f"/suche/?query=test")
    results = response.json()["results"]
    for result in results:
        code = result["occupation"]["code"]
        if code in accepted_additions:
            assert result["added"] == accepted_additions[code]
        else:
            assert result["added"] is None
