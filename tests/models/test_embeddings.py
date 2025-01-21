# pylint: disable=redefined-outer-name
import pytest
import torch
from spacy.lang.de import German

from publicplan.models.embeddings import (CachedEmbeddings, StringCache,
                                          SumWordsEmbedding, WordEmbedding)
from publicplan.nlp import Tokenizer


@pytest.fixture()
def cache_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("test_cache")


def test_check_vectors():
    with pytest.raises(ValueError):
        nlp = German()
        WordEmbedding("test", nlp)


def test_swe_vectorize(mock_embedding):

    text0 = "test, und, test"
    text1 = "test test"
    text2 = ", und "
    text3 = ""
    texts = [text0, text1, text2, text3]

    swe1 = SumWordsEmbedding(mock_embedding, Tokenizer(filter_stops=True))
    swe2 = SumWordsEmbedding(mock_embedding, Tokenizer(filter_stops=False))
    emb1 = swe1.vectorize(texts)
    emb2 = swe2.vectorize(texts)

    assert (emb1[0] != emb2[0]).any()
    assert (emb1[0] == emb2[1]).all()
    assert (emb1[2] == emb1[3]).all()


def test_cached_embs(descriptions, swe):
    fields = descriptions.fields
    codes = descriptions.codes
    cached_embs = CachedEmbeddings(descriptions, swe, fields=fields, ccr=False)

    desc = descriptions[codes[2]]
    string_cached, list_cached = cached_embs.get(codes)
    for field in descriptions.string_fields:
        emb = swe.vectorize([getattr(desc, field)])
        assert (emb[0] == string_cached[field][2]).all()
    for field in descriptions.list_fields:
        entries = getattr(desc, field)
        if not entries:
            assert (torch.zeros((0, swe.dim)) == list_cached[field][2]).all()
        else:
            emb = swe.vectorize(entries)
            assert (emb == list_cached[field][2]).all()


def test_cached_embs_ccr(descriptions, swe):

    fields = descriptions.fields
    codes = descriptions.codes
    cached_embs = CachedEmbeddings(descriptions, swe, fields=fields, ccr=True)

    desc = descriptions[codes[2]]
    string_cached, _ = cached_embs.get(codes)
    for field in string_cached:
        fp = cached_embs.first_principals[field]
        if field in descriptions.list_fields:
            emb = swe.vectorize(getattr(desc, field)).mean(0)
        else:
            emb = swe.vectorize([getattr(desc, field)])[0]
        emb_reduced = emb - emb.dot(fp) * fp
        assert (emb_reduced == string_cached[field][2]).all()


def test_string_cache(descriptions, swe):
    cache = StringCache()
    cache.update_collection(descriptions, swe)

    assert cache.data
    assert all(isinstance(h, int) for h in cache.data)
    assert all(emb.size() == (swe.dim, ) for emb in cache.data.values())


def test_string_cache_emb(descriptions, swe):

    cache = StringCache()
    cached_embs1 = CachedEmbeddings(descriptions, swe, cache=cache)
    cached_embs2 = CachedEmbeddings(descriptions, swe)

    for f in cached_embs1.string_data:

        data1 = cached_embs1.string_data[f]
        data2 = cached_embs2.string_data[f]
        assert (data1 == data2).all()

    for f in cached_embs1.list_data:

        data1 = cached_embs1.list_data[f]
        data2 = cached_embs2.list_data[f]
        assert len(data1) == len(data2)
        assert all((emb1 == emb2).all() for emb1, emb2 in zip(data1, data2))


def test_string_cache_load(descriptions, swe, cache_dir):
    cache1 = StringCache()
    cache1.update_collection(descriptions, swe)
    cache1.save(cache_dir)

    cache2 = StringCache.from_path(cache_dir)
    assert cache1.data.keys() == cache2.data.keys()
    assert all((cache1.data[h] == cache2.data[h]).all() for h in cache1.data)


def test_embedded_docs_simil(emb_simil, codes):

    code = codes[2]
    query = "Test"
    scores = emb_simil.get(query, codes=[code])[0]
    assert (scores >= -1).all()
    assert (scores <= 1).all()
    assert (scores == 0).any()
    assert (scores != 0).any()
