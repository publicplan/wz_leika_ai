from __future__ import annotations

from abc import ABC, abstractmethod, abstractproperty
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam, Optimizer  # type: ignore

from .features import Features, ScoreFeatures


class Classifier(ABC, nn.Module):
    """Base class for relevance classifiers."""
    @abstractproperty
    def features(self) -> Features:
        """Features required by model."""

    @abstractmethod
    def batch_predict(self,
                      queries: List[str],
                      num_returned: Optional[int] = None) -> List[pd.Series]:
        """Predict results for multiple queries.

        Args:
            queries: Queries to predict.
            num_returned: If not None, return only this many results.

        Returns:
            List of ordered relevance probabilities as pandas series,
            with codes as index, one for each query.
        """

    def predict(self,
                query: str,
                num_returned: Optional[int] = None) -> pd.Series:
        """Predict results for query.

        Args:
            query: Query to predict.
            num_returned: If not None, return only this many results.

        Returns:
            Ordered relevance probabilities as pandas series,
            indexed by the codes.
        """
        return self.batch_predict([query], num_returned=num_returned)[0]

    @abstractmethod
    def save_checkpoint(self, checkpoint: Union[Path, str]) -> None:
        """Save trained model to path."""

    @abstractmethod
    def describe_weights(self) -> Dict[str, float]:
        """Description of the model weights."""

    @property
    def device(self) -> torch.device:
        """Device model is loaded on."""
        return next(self.parameters()).device

    # pylint: disable=unused-argument
    def optimizer(self, lr: float, weight_decay: float = 0.0) -> Optimizer:
        """Optimizer to use for training.

        Args:
            lr: Learning rate.
        weight_decay: Weight decay.

        Returns:
            Initialized Optimizer.
        """
        return Adam(self.parameters(), lr=lr)

    def update_cache(self) -> None:
        """Update internal cache."""


class LinearClassifier(Classifier):
    """Simple logistic model for relevance classifications.

    Uses word similiarity and term-document scores as features.

    Args:
        features: Features to use for classification.
    """
    def __init__(self, features: ScoreFeatures):

        super().__init__()

        self._features = features

        input_dim = self._features.dim
        self.linear = nn.Linear(in_features=input_dim, out_features=1)
        nn.init.constant_(self.linear.weight, 1.)

    @property
    def features(self) -> ScoreFeatures:
        return self._features

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:  #type: ignore

        features = inputs[0]
        out = self.linear(features)

        return out.sigmoid()

    def batch_predict(self,
                      queries: List[str],
                      num_returned: Optional[int] = None) -> List[pd.Series]:
        self.eval()
        codes = self.features.codes
        features = self.features.batch_get(queries, codes)
        inputs = (feature.to(self.device) for feature in features)

        with torch.no_grad():
            preds = self(*inputs).squeeze().cpu().numpy()

        num_codes = len(codes)
        preds_split = [
            preds[i * num_codes:(i + 1) * num_codes]
            for i in range(len(queries))
        ]
        if num_returned is None:
            num_returned = len(codes)
        predictions = [
            pd.Series(pred,
                      index=codes).sort_values(ascending=False)[:num_returned]
            for pred in preds_split
        ]

        return predictions

    def describe_weights(self) -> Dict[str, float]:

        description = {
            desc: self.linear.weight[0, i].item()
            for i, desc in enumerate(self._features.describe())
        }
        description["Bias"] = self.linear.bias.item()

        return description

    def load_weights(self, weights_path: Union[Path, str]) -> None:
        """Load weights from path."""
        self.load_state_dict(torch.load(weights_path,
                                        map_location=self.device))

    def save_checkpoint(self, checkpoint: Union[Path, str]) -> None:
        weights_path = Path(checkpoint).joinpath("model.pt")
        torch.save(self.state_dict(), weights_path)
