import pytest
from pydantic import ValidationError

from publicplan.wz.description import WZCode, WZDesc


def test_code_parsing():

    code1 = WZCode.parse("10103")
    code2 = WZCode("10.10.3")

    assert code1 == code2
    assert int(code2) == 10103


def test_description():
    base_description = {
        "name": "test",
        "section_name": "test",
        "group_name": "test",
        "class_name": "test",
        "explanation": "test",
        "exclusions": "test",
        "keywords": []
    }

    desc1 = WZDesc(code="01.11.0", **base_description)
    desc2 = WZDesc(code="01110", **base_description)

    assert desc1 == desc2
    assert int(desc1) == 1110

    for wrong_code in ["1110", "abc", "12.12"]:
        with pytest.raises(ValidationError):
            WZDesc(code=wrong_code, **base_description)


def test_xml_parsing(wz_descs):

    expected_codes = [
        "01.11.0", "01.12.0", "01.13.1", "01.13.2", "01.14.0", "01.15.0"
    ]
    assert expected_codes == [desc.code for desc in wz_descs.values()]

    desc = wz_descs["01.11.0"]

    assert desc.name == "Anbau von Getreide (ohne Reis), Hülsenfrüchten und Ölsaaten"
    assert desc.section_name == "Landwirtschaft, Jagd und damit verbundene Tätigkeiten"
    assert desc.group_name == "Anbau einjähriger Pflanzen"

    assert len(desc.keywords) == 22
