import pandas as pd


# pylint: disable=protected-access
def test_termdoc_score(termdocs, codes):

    #pylint: disable=protected-access
    assert (termdocs._data.index == pd.Index(codes)).all()

    scores = termdocs.score("beglaubigung zahlung", codes)

    assert scores["method"][codes[0]] > 0
    assert scores["method"][codes[1]] > 0
    assert scores["method"][codes[2]] == 0
    assert (scores >= 0).all(axis=None)

    scores = termdocs.score("wohnraumhilfe", codes)
    assert scores["synonyms"][codes[2]] > 0

    scores = termdocs.score("adsdsadwwda", codes)

    assert (scores == 0).all(axis=None)
