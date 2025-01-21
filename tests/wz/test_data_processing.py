# pylint: disable=redefined-outer-name
import pytest
from publicplan.wz.data_processing import process_ihk, ihk_keywords


@pytest.fixture
def ihk_additions():

    return [("01.13.1.0", "Anbau v. Pilzen"),
            ("93.11.0.0", "Schwimmbad / Erlebnisbad"),
            ("93.11.0.1", "Kegelbahnen / Bowlingbahnen"),
            ("93.11.0.2", "Bootshaus"), ("93.11.0.3", "Tennishalle / Squash"),
            ("93.11.0.4", "Golfplatz"),
            ("93.11.0.5", "Eiskunstbahn / Wintersportstadien / Rollschuhbahn"),
            ("93.11.0.6", "Reithalle / Reitplatz"),
            ("93.11.0.7", "Sportanlagen, a.n.g. / z.B. Fußballstadien"),
            ("93.11.0.8", "Minigolfanlagen")]


def test_ihk(ihk_descs, ihk_additions):
    ihk_names = process_ihk(ihk_descs, expand_abbrev=False)
    sections = [
        ("01", "Landwirtschaft, Jagd u. damit verbundene Tätigkeiten"),
        ("93",
         "Erbringung v. Dienstleistungen des Sports, der Unterhaltung u. der Erholung"
         )
    ]
    groups = [("01.1", "Anbau einjähriger Pflanzen"),
              ("93.1", "Erbringung v. Dienstleistungen des Sports")]
    classes = [
        ("01.12", "Anbau v. Reis"), ("93.11", "Betrieb v. Sportanlagen"),
        ("01.13", "Anbau v. Gemüse u. Melonen sowie Wurzeln u. Knollen")
    ]
    subclasses = [
        ("01.11.0",
         "Anbau v. Getreide (ohne Reis), Hülsenfrüchten u. Ölsaaten"),
        ("01.12.0", "Anbau v. Reis"),
        ("01.13.1", "Anbau v. Gemüse u. Melonen"),
        ("93.11.0", "Betrieb v. Sportanlagen")
    ]

    assert all(s in ihk_names["sections"] for s in sections)
    assert all(g in ihk_names["groups"] for g in groups)
    assert all(c in ihk_names["classes"] for c in classes)
    assert all(sc in ihk_names["subclasses"] for sc in subclasses)
    assert all(a in ihk_names["additions"] for a in ihk_additions)


def test_ihk_expand_abbrev(ihk_descs):
    ihk_expanded = process_ihk(ihk_descs, expand_abbrev=True)
    abbrev = ("01", "Landwirtschaft, Jagd u. damit verbundene Tätigkeiten")
    expanded = ("01", "Landwirtschaft, Jagd und damit verbundene Tätigkeiten")
    assert expanded in ihk_expanded["sections"]
    assert abbrev not in ihk_expanded["sections"]

    ihk_abbrev = process_ihk(ihk_descs, expand_abbrev=False)
    assert expanded not in ihk_abbrev["sections"]
    assert abbrev in ihk_abbrev["sections"]


def test_ihk_keywords(ihk_descs, ihk_additions):
    ihk_kws = ihk_keywords(ihk_descs)

    assert set(ihk_kws.keys()) == {"01.13.1", "93.11.0"}

    for code, add in ihk_additions:
        add = add.replace("v.", "von")
        assert add in ihk_kws[code[:7]]

    assert not ihk_kws["11.11.1"]
