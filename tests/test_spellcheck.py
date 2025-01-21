# pylint: disable=redefined-outer-name
import pytest

from publicplan.paths import MOCK_DATA_DIR
from publicplan.spellcheck import SpellChecker


@pytest.fixture
def spellcheck():
    return SpellChecker(MOCK_DATA_DIR.joinpath("wordfreq.txt"),
                        edit_distance=2)


def test_spellspellcheck(spellcheck):
    assert spellcheck.correct("den") == "der"
    assert spellcheck.correct("Den") == "Der"
    assert spellcheck.correct("das") == "die"

    assert spellcheck.correct("nächte") == "nicht"
    assert spellcheck.correct("nächten") == "nächten"
    assert spellcheck.correct("den:das/der nächte.") == "der:die/der nicht."

    assert spellcheck.correct("DAS") == "DAS"
    assert spellcheck.correct("8den") == "8den"
    assert spellcheck.correct("den8") == "der"
