# pylint: disable=redefined-outer-name,undefined-variable
from urllib.parse import quote_plus

import pytest
from starlette.testclient import TestClient

from publicplan.leika.api import build_leika_api
from publicplan.leika.description import LeikaDesc


@pytest.fixture
def app(mock_clf, raw_descriptions, mock_spellcheck):

    app = build_leika_api(mock_clf, raw_descriptions, raw_descriptions,
                          mock_spellcheck)

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


def test_parsing(client):

    query = r"/a b \ \\ \n* . $ § '`'%20 %2F %"
    encoded = quote_plus(query)
    response = client.get(f"/parse/?query={encoded}")
    assert response.status_code == 200

    assert response.json() == query


def test_api_query(client):

    query = 'Falsches Üben von Xylophonmusik quält jeden größeren Zwerg'
    query += r"/a b * . $ § & ?'`'%20 %2F %"
    encoded = quote_plus(query)
    num_results = 4
    response = client.get(
        f"/suche/?query={encoded}&num_returned={num_results}")

    assert response.status_code == 200

    response_data = response.json()
    assert len(response_data) == num_results

    #rounding error tolerance
    eps = 0.001

    for result in response_data:
        # test description parsing
        assert LeikaDesc.from_dict(result['service'])

        p = result['relevance-confidence']
        assert p >= -eps
        assert p <= 1 + eps

    assert response_data == sorted(response_data,
                                   key=lambda r: r["relevance-confidence"],
                                   reverse=True)


def test_api_codes(client, raw_descriptions):

    response = client.get("/codes")

    assert response.status_code == 200
    assert response.json() == raw_descriptions


def test_api_rejected(client, raw_descriptions):

    response = client.get("/rejected")

    assert response.status_code == 200
    assert response.json() == raw_descriptions


def test_api_descs(client, raw_descriptions, codes):
    code = codes[0]

    response = client.get(f"/descs/{code}")
    assert response.status_code == 200

    expected = None
    for desc in raw_descriptions:
        if int(desc["schluessel"]) == code:
            expected = desc
            break

    assert expected
    assert response.json() == expected


def test_spellcheck(client):

    query = "Test"
    encoded = quote_plus(query)
    response = client.get(f"/spellcheck/?query={encoded}")

    assert response.status_code == 200

    assert response.json() == query
