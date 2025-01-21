from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from textacy.vsm.vectorizers import Vectorizer

from publicplan.documents import Collection
from publicplan.nlp import Tokenizer

from .features import ScoreFeatures


class TermDocs(ScoreFeatures):
    """Class for building term-document matrices for document collections.

    Provides features through weighted and normalized term-document scores.

    Args:
        collection: Document collection to use.
        fields: Which fields of the collection to use. If None, use all
            available fields.
        tokenizer: Tokenizer to use for splitting and filtering.
            If None, use default Tokenizer.
        tf_type: Type of term frequency weighting.
            One of 'linear', 'sqrt', 'log' or 'binary'
        idf_type: Type of inverse document frequency weighting.
            One of 'standard', 'smooth', 'bm25' or None.
            If None, do not weight by document frequency
        norm: Weight normalization of doc vectors.
            One of None, 'l1' or 'l2'
    """
    def __init__(self,
                 collection: Collection,
                 fields: Optional[List[str]] = None,
                 tokenizer: Optional[Tokenizer] = None,
                 tf_type: str = "linear",
                 idf_type: Optional[str] = "standard"):

        self.tokenizer = tokenizer if tokenizer else Tokenizer()
        self.tf_type = tf_type
        self.idf_type = idf_type
        self._norm = "l2"
        self._codes = collection.codes

        if fields is None:
            self.fields = collection.fields
        else:
            self.fields = fields

        self._vocabs: Dict[str, Dict[str, int]] = {}

        self._data = self._build_termdocs(collection)

    @property
    def dim(self) -> int:
        return len(self.fields)

    @property
    def codes(self) -> List[int]:
        return self._codes

    def score(self, text: str, codes: List[int]) -> pd.DataFrame:
        """Sum up term weights for terms appearing in text.

        Args:
            text: Text to score.
            codes: Subset of codes to consider.

        Returns:
            Scores for each field and given codes as dataframe.
        """

        scores: List[pd.DataFrame] = []
        for field in self.fields:
            vocab = self._vocabs[field]
            terms = [t for t in self._to_terms_list(text) if t in vocab.keys()]

            if terms == []:
                scores.append(pd.Series(0.0, index=codes, dtype=np.float32))
                continue

            idx = [vocab[term] for term in terms]
            term_scores = self._data[field].reindex(index=codes, columns=idx)

            field_scores = term_scores.sum(axis=1) / np.sqrt(len(terms))
            scores.append(field_scores)

        return pd.concat(scores, axis=1, keys=self.fields)

    def _build_termdocs(self, collection: Collection) -> pd.DataFrame:

        termdocs: List[pd.DataFrame] = []
        for field in self.fields:

            vectorizer = self._build_vectorizer()

            entries: List[str]
            if field in collection.list_fields:
                entries = [
                    " ".join(entry) for entry in collection.list_entries(field)
                ]
            else:
                entries = collection.string_entries(field)
            tokenized_docs = [
                self.tokenizer.split(entry.lower()) for entry in entries
            ]
            termdoc_matrix = vectorizer.fit_transform(tokenized_docs)

            index = pd.Index(self.codes)
            #Using sparse matrices here slows down retrieval by a factor of 10.
            termdoc = pd.DataFrame(termdoc_matrix.todense(),
                                   index=index,
                                   dtype=np.float32)
            # termdoc = pd.DataFrame.sparse.from_spmatrix(termdoc_matrix,
            #                                             index=index)

            termdocs.append(termdoc)
            self._vocabs[field] = vectorizer.vocabulary_terms

        return pd.concat(termdocs, axis=1, keys=self.fields)

    def _build_vectorizer(self) -> Vectorizer:
        if self.idf_type is None:
            return Vectorizer(tf_type=self.tf_type,
                              apply_idf=False,
                              norm=self._norm)

        return Vectorizer(tf_type=self.tf_type,
                          apply_idf=True,
                          idf_type=self.idf_type,
                          norm=self._norm)

    def _to_terms_list(self, doc: Union[str, List[str]]) -> List[str]:
        if isinstance(doc, list):
            doc = " ".join(doc)

        return self.tokenizer.split(doc.lower())

    def get(self, query: str, codes: List[int]) -> Tuple[torch.Tensor]:

        # pylint: disable=not-callable
        return (torch.tensor(self.score(query, codes).to_numpy()), )

    def describe(self) -> List[str]:

        desc = "Term-doc score for "
        return [desc + field for field in self.fields]
