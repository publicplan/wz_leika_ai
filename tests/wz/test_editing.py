from publicplan.wz.editing import WZAddition, parse_additions, update_descs


def test_update(wz_descs, accepted_additions):

    invalid = WZAddition(code="99.99.9")
    additions = list(accepted_additions.values()) + [invalid]
    new_descs, accepted, rejected = update_descs(wz_descs, additions)

    assert rejected == [invalid]
    assert accepted == accepted_additions

    assert new_descs != wz_descs

    for code in new_descs:
        desc = new_descs[code].dict()
        old_desc = wz_descs[code].dict()
        if desc["code"] in accepted_additions:
            add = accepted_additions[desc["code"]].dict()

            # assert desc["code"] == old_desc["code"]
            assert all(desc[field] == " ".join([old_desc[field], add[field]])
                       for field in [
                           "name", "section_name", "group_name", "class_name",
                           "explanation"
                       ])
            assert desc["keywords"] == old_desc["keywords"] + add["keywords"]
        else:
            assert desc == old_desc


def test_parse_additions(accepted_additions, rejected_additions):

    malformed_addition = {"abc": 0}
    accepted = list(accepted_additions.values())
    raw_adds = [acc.dict() for acc in accepted
                ] + rejected_additions + [malformed_addition]
    parse_accepted, parse_rejected = parse_additions(raw_adds)

    assert parse_accepted == accepted
    assert parse_rejected == rejected_additions + [malformed_addition]
