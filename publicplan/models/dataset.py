from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Callable, List, Optional, Set, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from .features import Features

logger = logging.getLogger(__name__)


@dataclass
class QueryData:
    """Class for encoding test query and its relevant codes.

    Args:
        query: Test query
        relevant: Codes judged relevant for this query.
    """
    query: str
    relevant: Set[int]


class SampledQueryData(Dataset):
    """Dataset class for generating samples from query data.

    Each item is constructed by sampling relevant and irrelevant codes for
    the given query. A batch is a tuple of arrays calculated from the labels
    and features of the sampled codes.
    The first entry are the labels, while the following entries are the features

    Args:
        data: Query dataset to use.
        features: Features to extract from data and document descriptions.
        sample_size: Number of codes to sample for each query.
        prefilter_fn: If not None, use this function to narrow down codes before sampling.
    """
    def __init__(self,
                 data: List[QueryData],
                 features: Features,
                 sample_size: int,
                 prefilter_fn: Optional[Callable[[str], List[int]]] = None):

        self.data = data
        self.features = features
        self.sample_size = sample_size

        if prefilter_fn is None:
            self.prefilter_fn = lambda _: self.features.codes
        else:
            self.prefilter_fn = prefilter_fn

    def __len__(self) -> int:

        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, ...]:

        qd = self.data[index]
        prefiltered = set(self.prefilter_fn(qd.query))
        irrelevant = prefiltered - qd.relevant
        sample_pos, sample_neg = _sample(qd.relevant,
                                         irrelevant,
                                         size=self.sample_size)

        labels = torch.zeros((len(sample_pos) + len(sample_neg), 1))
        labels[:len(sample_pos)] = 1.0

        inputs = self.features.get(qd.query, sample_pos + sample_neg)

        return (labels, ) + inputs

    def collate_fn(
            self, samples: List[Tuple[torch.Tensor,
                                      ...]]) -> Tuple[torch.Tensor, ...]:
        """Combine multiple samples to a training batch."""

        labels = torch.cat([sample[0] for sample in samples], dim=0)
        inputs = self.features.collate_fn([sample[1:] for sample in samples])

        return (labels, ) + inputs


def _sample(relevant_codes: Set[int], irrelevant_codes: Set[int],
            size: int) -> Tuple[List[int], List[int]]:
    size_pos = min(len(relevant_codes), int(size / 2))
    sample_pos = random.sample(relevant_codes, k=size_pos)

    size_neg = min(size - size_pos, len(irrelevant_codes))
    sample_neg = random.sample(irrelevant_codes, k=size_neg)

    return sample_pos, sample_neg


def get_dataloader(data: List[QueryData],
                   features: Features,
                   sample_size: int,
                   queries_per_batch: int,
                   prefilter_fn: Optional[Callable[[str], List[int]]] = None,
                   num_workers: int = 0,
                   drop_last: bool = True) -> DataLoader:
    """Build dataloader for given dataset.

    The dataloader yields batches of features and labels, which are constructed
    by concatenating sampled codes from queries in dataset.

    Args:
        data: Query dataset to use.
        features: Features to extract from data and document descriptions.
        sample_size: Number of codes to sample for each query.
        queries_per_batch: Number of queries sampled for each batch.
        drop_last: Whether to drop the last, incomplete batch.
        prefilter_fn: If not None, use this function to narrow down codes before sampling.

    Returns:
        Dataloader yielding batches of sampled features and labels.
    """

    sampled = SampledQueryData(data,
                               features,
                               sample_size=sample_size,
                               prefilter_fn=prefilter_fn)

    return DataLoader(sampled,
                      batch_size=queries_per_batch,
                      collate_fn=sampled.collate_fn,
                      num_workers=num_workers,
                      drop_last=drop_last)
