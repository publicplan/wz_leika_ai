from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
from abc import abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from zipfile import ZipFile

import numpy as np
import spacy
import torch
import torch.nn.functional as F
from spacy.lang.de import German

from publicplan.documents import Collection
from publicplan.nlp import Tokenizer
from publicplan.paths import EMBEDDINGS_DIR

from . import utils
from .features import ScoreFeatures

logger = logging.getLogger(__name__)


class WordEmbedding:
    """Wrapper class for word embedding models.

        Args:
            name: Identifier for the model.
            nlp: spacy language model with word vectors.
            subword_range: Length of subwords to consider when word is oov.

        Raises:
            ValueError: When the language model does not contain word vectors.
    """
    def __init__(self,
                 name: str,
                 nlp: German,
                 subword_range: Optional[Tuple[int, int]] = None):

        self.nlp = nlp
        self.name = name

        self.vector_dim = self.nlp.meta["vectors"]["width"]
        if self.vector_dim == 0:
            raise ValueError(
                "Language model for WordEmbeding needs word vectors")

        self._subword_range = subword_range

        self._minn: Optional[int] = None
        self._maxn: Optional[int] = None
        if subword_range:
            self._minn, self._maxn = subword_range

    @classmethod
    def from_path(
            cls,
            name: str,
            weights_path: Optional[Union[Path, str]] = None,
            subword_range: Optional[Tuple[int, int]] = None) -> WordEmbedding:
        """Load model from path.

        The weights path should point to a directory created by spacy's
        init-model function.

        Args:
            name: Identifier for the model.
            weights_path: Path to weights directory. If this is none, look
                for name in the default weights directory for embeddings.
            subword_range: Length of subwords to consider when word is oov.
        """
        if weights_path is None:
            weights_path = EMBEDDINGS_DIR.joinpath(name)

        weights_path = Path(weights_path)
        if not weights_path.exists():
            zip_path = weights_path.with_suffix(".zip")
            if zip_path.exists():
                with ZipFile(zip_path, "r") as zf:
                    zf.extractall(EMBEDDINGS_DIR)

        nlp = spacy.load(weights_path)
        return cls(name, nlp, subword_range=subword_range)

    def vectorize(self, word: str, normalize: bool = False) -> np.ndarray:
        """Computes vector representation of text.

        Input is interpreted as word. If word is oov, use subwords if
        model provides them. Otherwise returns 0.
        """
        if self.is_oov(word) and not self.is_oov(word.capitalize()):
            word = word.capitalize()
        v = self.nlp.vocab.get_vector(word, minn=self._minn, maxn=self._maxn)

        if normalize:
            norm = np.sqrt(v.T.dot(v))
            if norm < 1e-8:
                v = np.zeros_like(v)
            else:
                v = v / norm

        return v

    def is_oov(self, word: str) -> bool:
        """Check wether word is out-of-vocabulary (oov)."""
        return self.nlp(word)[0].is_oov  # type: ignore


class Embedding:
    """Interface for generic text embeddings."""
    @abstractmethod
    def vectorize(self, texts: List[str]) -> torch.Tensor:
        """Compute embedding of list of texts.

        Returns results stacked along dimension zero.
        """

    @property
    @abstractmethod
    def dim(self) -> int:
        """Dimension of embedding vectors."""

    @property
    def device(self) -> torch.device:
        """Device of output vectors."""
        return torch.device("cpu")


class SumWordsEmbedding(Embedding):
    """CBOW doc embedding, based on specified word embedding.

    Args:
        we: Word embedding model.
        tokenizer: Tokenizer to use. If None, use default tokenizer.
        normalize: Whether to normalize word embeddings before summing.
    """
    def __init__(self,
                 we: WordEmbedding,
                 tokenizer: Optional[Tokenizer] = None,
                 normalize: bool = True):

        self.we = we
        self.normalize = normalize
        self.tokenizer = tokenizer if tokenizer else Tokenizer()

    def vectorize(self, texts: List[str]) -> torch.Tensor:
        """Averages word vectors of tokens contained in each text.
        Stacks resulting embedding along dimension zero.
        """

        result: List[torch.Tensor] = []
        for text in texts:
            tokens = self.tokenizer.split(text)

            if not tokens:
                result.append(
                    torch.zeros(self.we.vector_dim,
                                dtype=torch.float32,
                                device=self.device))
                continue
            # pylint: disable=not-callable
            embs = [
                torch.tensor(self.we.vectorize(token, self.normalize))
                for token in tokens
            ]
            result.append(torch.stack(embs).mean(dim=0))

        return torch.stack(result)

    @property
    def dim(self) -> int:
        return self.we.vector_dim  # type: ignore


@dataclasses.dataclass
class StringCache:
    """Class for catching text embeddings.

    Args:
        data: Embedding dictionary indexed by text hashes.
    """

    data: Dict[int, torch.Tensor] = dataclasses.field(default_factory=dict)

    def get(self, text: str) -> torch.Tensor:
        """Get stored embedding of text.

        Raises KeyError if text was not cached.
        """
        return self.data[string_hash(text)]

    def update(self, texts: List[str], embedding: Embedding) -> None:
        """Update cache with given texts using embedding."""
        not_cached = {
            string_hash(t): t
            for t in sorted(texts, key=len) if string_hash(t) not in self.data
        }
        if not not_cached:
            return
        hashes = list(not_cached.keys())
        batch_size = max(len(hashes) // 10, 100)
        logger.info(f"Updating cache. Computing %s new embeddings.",
                    len(hashes))
        for n in range(0, len(hashes), batch_size):
            hash_batch = hashes[n:n + batch_size]
            texts = [not_cached[h] for h in hash_batch]
            embs = embedding.vectorize(texts)
            for i, h in enumerate(hash_batch):
                self.data[h] = embs[i]
            logger.info("Computed %s/%s embeddings.",
                        min(len(hashes), n + batch_size), len(hashes))

    def update_collection(self,
                          collection: Collection,
                          embedding: Embedding,
                          fields: Optional[List[str]] = None) -> None:
        """Update cache with texts from collection.

        Args:
            collection: Document collection to extract texts from.
            embedding: Embedding to use for computation.
            fields: Fields to cache.
        """
        if fields is None:
            fields = collection.fields
        texts: List[str] = []
        string_fields = [f for f in fields if f in collection.string_fields]
        list_fields = [f for f in fields if f in collection.list_fields]
        for field in string_fields:
            texts += collection.string_entries(field)
        for field in list_fields:
            texts += [
                s for entry in collection.list_entries(field) for s in entry
            ]
        self.update(texts, embedding)

    @classmethod
    def from_path(cls,
                  path: Union[Path, str],
                  device: Optional[torch.device] = None) -> StringCache:
        """Load cache from path.

        Args:
            path: Directory with stored cache.
            device: Torch device to load embeddings on. If None, use cpu.
        Returns:
            Loaded Cache.
        """

        if device is None:
            device = torch.device("cpu")
        path = Path(path)
        hashes = json.load(open(path.joinpath("cache_hashes.json")))
        embs = torch.load(path.joinpath("cache.pt"), map_location=device)
        data = dict(zip(hashes, embs))
        return cls(data)

    def save(self, path: Union[Path, str]) -> None:
        """Save cache to directory.

        Args:
            path: Directory to store cache in.
        """
        hashes = list(self.data.keys())
        embs = torch.stack([self.data[h] for h in hashes])
        path = Path(path)
        json.dump(hashes, open(path.joinpath("cache_hashes.json"), "w"))
        torch.save(embs, path.joinpath("cache.pt"))


class CachedEmbeddings:
    """Cache embeddings for documents in collection.

    Args:
        collection: Collection to cache.
        embedding: Embedding to use.
        fields: Which fields to cache. If None, use all available.
        ccr: If true, for each field remove the first principal component from
            the corresponding embedding matrix.
        cache: Cache of string embedding to use for faster loading.
    """
    def __init__(self,
                 collection: Collection,
                 embedding: Embedding,
                 fields: Optional[List[str]] = None,
                 ccr: bool = False,
                 cache: Optional[StringCache] = None):

        self.collection = collection
        self.embedding = embedding
        if fields is None:
            fields = collection.fields
        self._string_fields = [
            f for f in fields if f in collection.string_fields
        ]
        self._list_fields = [f for f in fields if f in collection.list_fields]
        self.ccr = ccr
        self.codes = self.collection.codes

        self.first_principals: Dict[str, torch.Tensor] = {}

        self._string_data: Dict[str, torch.Tensor] = {}
        self._list_data: Dict[str, List[torch.Tensor]] = {}
        if cache is not None:
            cache.update_collection(collection, embedding, fields=fields)
        self._compute_string_data(cache)
        self._compute_list_data(cache)

    def _compute_embeddings(self, texts, cache: Optional[StringCache] = None):
        if cache is None:
            return self.embedding.vectorize(texts)
        cached = torch.stack([cache.data[string_hash(t)] for t in texts])
        return cached

    @property
    def string_data(self) -> Dict[str, torch.Tensor]:
        return self._string_data

    def _compute_string_data(self,
                             cache: Optional[StringCache] = None) -> None:

        for field in self._string_fields:
            entries = self.collection.string_entries(field)
            data = self._compute_embeddings(entries, cache)
            if self.ccr:
                data, first_principal = utils.ccr(data)
                self.first_principals[field] = first_principal
            self._string_data[field] = data

    @property
    def list_data(self) -> Dict[str, List[torch.Tensor]]:
        return self._list_data

    def _compute_list_data(self, cache: Optional[StringCache] = None) -> None:

        for field in self._list_fields:
            field_data: List[torch.Tensor] = []
            for entry in self.collection.list_entries(field):
                if not entry:
                    embs = torch.zeros((1, self.embedding.dim),
                                       dtype=torch.float32,
                                       device=self.embedding.device)
                else:
                    embs = self._compute_embeddings(entry, cache)
                field_data.append(embs)

            pooled = torch.stack([embs.mean(dim=0) for embs in field_data])
            if self.ccr:
                pooled, fp = utils.ccr(pooled)
                self.first_principals[field] = fp
                for n, embs in enumerate(field_data):
                    field_data[n] = embs - embs.mm(
                        fp.unsqueeze(1)).squeeze(1).ger(fp)
            self._string_data[field] = pooled
            self._list_data[field] = field_data

    def get(
        self, codes: List[int]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]]]:
        """Get cached embeddings for given codes.

        Args:
            codes: Codes to return embeddings for.

        Returns:
            String and list embeddings in dictionary, indexed by field.
        """
        ids = [self.collection.inverted_index[code] for code in codes]

        string_data = {f: data[ids] for f, data in self.string_data.items()}
        list_data = {
            f: [data[id] for id in ids]
            for f, data in self.list_data.items()
        }

        return string_data, list_data


def string_hash(s: str) -> int:
    h = hashlib.blake2b(s.encode(), digest_size=8)
    return int(h.hexdigest(), base=16)


class EmbSimil(ScoreFeatures):
    """Class for computing and storing embeddings for document collections.

    Provides features based on cosine similarity.

    The first common component of the field matrix is subtracted. See

        Arora et. al: A simple but tough-to-beat baseline for sentence embeddings,

    for details.

    Args:
        cache: Cached embeddings for collection
        list_maxpool: If true, provide additionally score features by max pooling
            over list items.
    """
    def __init__(self, cache: CachedEmbeddings, list_maxpool=False):

        self.cache = cache
        self._embedding = self.cache.embedding
        self.ccr = self.cache.ccr
        self._string_fields = list(self.cache.string_data.keys())
        if list_maxpool:
            self._list_fields = list(self.cache.list_data.keys())
        else:
            self._list_fields = []

        self._list_block_size = 100

    @property
    def codes(self) -> List[int]:
        return self.cache.codes

    @property
    def dim(self) -> int:
        return len(self._string_fields) + len(self._list_fields)

    def get(self, query: str, codes: List[int]) -> Tuple[torch.Tensor, ...]:
        return self.batch_get([query], codes)

    def batch_get(self, queries: List[str],
                  codes: List[int]) -> Tuple[torch.Tensor, ...]:
        query_embs = self._embedding.vectorize(queries)
        scores = torch.cat([
            self._compute_scores(query_embs[i], codes)
            for i in range(query_embs.size(0))
        ])

        return (scores, )

    def _compute_scores(self, query_emb: torch.Tensor,
                        codes: List[int]) -> torch.Tensor:

        scores: List[torch.Tensor] = []
        string_data, list_data = self.cache.get(codes)
        for field in self._string_fields:
            sdata = string_data[field]
            fp = None if not self.ccr else self.cache.first_principals[field]
            score = _simil(query_emb, sdata, fp)
            scores.append(score)

        num_blocks = len(codes) // self._list_block_size
        for field in self._list_fields:
            ldata = list_data[field]

            # Batch process list data:
            # Create batches with similar size, to avoid
            # unnecessary padding and memory consumption
            ixs = list(range(len(ldata)))
            # pylint: disable=cell-var-from-loop
            ixs.sort(key=lambda i: ldata[i].size(0))
            inv_ixs = _order_inverse(ixs)
            ldata = [ldata[i] for i in ixs]
            block_scores: List[torch.Tensor] = []
            fp = None if not self.ccr else self.cache.first_principals[field]
            for n in range(num_blocks):
                block_embs = ldata[n * self._list_block_size:(n + 1) *
                                   self._list_block_size]
                block_embs = utils.pad_tensors(block_embs, dims=[0])
                score = _simil(query_emb, torch.stack(block_embs), fp)
                score = score.max(dim=1)[0]
                block_scores.append(score)
            rest = ldata[num_blocks * self._list_block_size:]
            if rest:
                rest = utils.pad_tensors(rest, dims=[0])
                score = _simil(query_emb, torch.stack(rest), fp)
                score = score.max(dim=1)[0]
                block_scores.append(score)
            field_scores = torch.cat(block_scores)[inv_ixs]

            scores.append(field_scores)
        return torch.stack(scores, dim=1)

    def describe(self) -> List[str]:

        desc = "Similarity score for "
        string_descs = [desc + f for f in self._string_fields]
        list_descs = [desc + f + " (MaxPool)" for f in self._list_fields]

        return string_descs + list_descs


def _simil(query_emb: torch.Tensor, embs: torch.Tensor,
           fp: Optional[torch.Tensor]) -> torch.Tensor:
    if fp is not None:
        query_emb = query_emb - query_emb.dot(fp) * fp
    query_emb = query_emb.expand_as(embs)
    score = F.cosine_similarity(embs, query_emb, dim=-1)

    return score


def _order_inverse(ixs: List[int]) -> List[int]:
    inv_dict = dict((ix, i) for i, ix in enumerate(ixs))
    inv = [ixs[inv_dict[i]] for i in range(len(ixs))]

    return inv
