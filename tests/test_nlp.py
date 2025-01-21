# pylint: disable=redefined-outer-name
import pytest

from publicplan.nlp import Lemmatizer, Tokenizer, build_lemma_dict
from publicplan.paths import MOCK_DATA_DIR


@pytest.fixture
def lemmatizer():

    return Lemmatizer(MOCK_DATA_DIR.joinpath("lemmatizer.json"))


def test_lemmatize(lemmatizer):
    assert lemmatizer.lemmas("Läufst") == ["laufen"]
    assert lemmatizer.lemmas("gehen") == []
    assert set(lemmatizer.lemmas("Sitz")) == {"sitzen", "Sitz"}

    assert lemmatizer.lemma("gehen") == "gehen"
    assert lemmatizer.lemma("sitz") in ["sitzen", "Sitz"]

    assert lemmatizer.lemma("") == ""


def test_lemma_dict():

    lemma_dict = build_lemma_dict(MOCK_DATA_DIR.joinpath("lemmatizer.json"))
    expected = {
        "sitz": ["sitzen", "Sitz"],
        "läufst": ["laufen"],
        "fahrräder": ["Fahrrad"]
    }
    assert lemma_dict == expected


def test_tokenizer(lemmatizer, monkeypatch):

    s = "test, -und test"
    tokenizer1 = Tokenizer(filter_connecting=True, filter_stops=False)
    tokenizer2 = Tokenizer(filter_connecting=True, filter_stops=True)
    tokenizer3 = Tokenizer(filter_connecting=False)

    assert tokenizer1.split(s) == ["test", "und", "test"]
    assert tokenizer2.split(s) == ["test", "test"]
    assert tokenizer3.split(s) == ["test", "-und", "test"]

    monkeypatch.setattr(tokenizer1, "_lemmatizer", lemmatizer)

    assert tokenizer1.split("Du Läufst") == ["Du", "laufen"]
    assert tokenizer1.split("Du gehst") == ["Du", "gehst"]
