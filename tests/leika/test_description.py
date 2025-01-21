#pylint: disable=redefined-outer-name
import pytest

from publicplan.leika.description import LeikaDesc


@pytest.fixture
def test_desc():
    return {
        "schluessel": "99001003000000",
        "gruppierung": "Abfall",
        "kennung": "Bioabfall",
        "verrichtung": "",
        "verrichtungsdetail": "",
        "bezeichnung": "Bioabfall",
        "bezeichnung2": "",
        "typ": "2/3",
        "datum": "15.05.2013 10:35",
        "besondere_merkmale": "Müllabfuhr",
        "synonyme": "Abfall|Abfallbeseitigung|Brennholz",
    }


def test_parse_desc(test_desc):

    ld = LeikaDesc.from_dict(test_desc)

    assert ld.code == 99001003000000
    assert ld.name == "Bioabfall"
    assert ld.synonyms == "Abfall Abfallbeseitigung Brennholz".split(" ")
    assert ld.other == ["Müllabfuhr"]

    with pytest.raises(KeyError):
        desc = test_desc.copy()
        desc.pop("kennung")
        LeikaDesc.from_dict(desc)

    with pytest.raises(ValueError):
        desc = test_desc.copy()
        desc["schluessel"] = "abc"
        LeikaDesc.from_dict(desc)


def test_descriptions(descriptions, codes):

    indices = [
        descriptions.inverted_index[code] for code in [codes[1], codes[0]]
    ]
    assert indices == [1, 0]
    assert all(descriptions.string_entries("name"))
    assert any(descriptions.list_entries("synonyms"))
    assert not all(descriptions.list_entries("synonyms"))
