from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple

import torch


class Features(ABC):
    """Base class for features build from a query and a collection of documents."""
    @abstractmethod
    def collate_fn(
            self, batches: List[Tuple[torch.Tensor,
                                      ...]]) -> Tuple[torch.Tensor, ...]:
        """Collate list of feature batches into single batch."""

    @property
    @abstractmethod
    def codes(self) -> List[int]:
        """Codes of documents."""

    @abstractmethod
    def get(self, query: str, codes: List[int]) -> Tuple[torch.Tensor, ...]:
        """Get features for combination of query and each code in codes."""

    def batch_get(self, queries: List[str],
                  codes: List[int]) -> Tuple[torch.Tensor, ...]:
        """Compute features for list of queries and given codes.

        Features are concatenated along dimension zero.
        """
        # Provide default implementation with self.collate_fn, intendended to be
        # overriden by select features where batching is more efficient.
        features = [self.get(query, codes) for query in queries]
        return self.collate_fn(features)


class ScoreFeatures(Features, ABC):
    """Base class for features based on matching scores for query and documents."""
    @abstractmethod
    def describe(self) -> List[str]:
        """Describe returned score vector.

        Returns:
            List of strings, describing the various scores.
        """

    @property
    @abstractmethod
    def dim(self) -> int:
        """Dimension of features."""

    def collate_fn(
            self, batches: List[Tuple[torch.Tensor,
                                      ...]]) -> Tuple[torch.Tensor, ...]:
        """Collate list of feature batches into single batch."""
        n = len(batches[0])
        return tuple(
            torch.cat([batch[i] for batch in batches], dim=0)
            for i in range(n))


class ConcatFeatures(ScoreFeatures):
    """Class for concatening multiple scoring features.

    Args:
        feature_list: List of features to concatenate

    Raises:
        ValueError, if the document collections of the underlying
        features do not agree.
    """
    def __init__(self, feature_list: List[ScoreFeatures]):

        f0 = feature_list[0]
        for f in feature_list[1:]:
            if f.codes != f0.codes:
                raise ValueError(
                    "Features must be based on the same document collection.")

        self.feature_list = feature_list
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        self._dim = sum(f.dim for f in self.feature_list)

    def get(self, query: str, codes: List[int]) -> Tuple[torch.Tensor]:

        result = [
            f.get(query, codes)[0].to(self.device) for f in self.feature_list
        ]

        return (torch.cat(result, dim=1), )

    def describe(self) -> List[str]:

        return sum([f.describe() for f in self.feature_list], [])

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def codes(self) -> List[int]:
        return self.feature_list[0].codes
