from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from inspect import signature
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

import torch

from publicplan.documents import Collection
from publicplan.nlp import Tokenizer
from publicplan.paths import WEIGHTS_DIR

from .bert import (BertFeatures, BertFinetuning, BertPool, BertTokenizer,
                   validate_checkpoint)
from .classifiers import Classifier, LinearClassifier
from .embeddings import (CachedEmbeddings, Embedding, EmbSimil, StringCache,
                         SumWordsEmbedding, WordEmbedding)
from .features import ConcatFeatures, ScoreFeatures
from .termdoc import TermDocs

logger = logging.getLogger(__name__)

C = TypeVar("C", bound="_ConfigMixin")  #pylint: disable=invalid-name


class _ConfigMixin(ABC):
    """Mixin for configuration data."""
    config_name = "config"

    @classmethod
    def from_path(cls: Type[C], path: Path) -> C:
        """Load configuration from path."""

        config = json.load(open(path, "r"))
        return cls.from_dict(config)

    @classmethod
    def from_dir(cls: Type[C], directory: Optional[Union[Path, str]]):
        """Load configuration from directory.

        Looks for config file with filename <name>.json in directory.
        Loads default value if directory is none or file does not exist.
        """

        if directory is None:
            return cls()

        directory = Path(directory)
        path = directory.joinpath(cls.config_name + ".json")
        if not path.exists():
            logger.warning(
                f"File {cls.config_name}.json not found in {directory.absolute()}."
            )
            logger.warning(f"Using default {cls.config_name} configuration.")
            return cls()

        logger.info(f"Using {cls.config_name} config from {directory}.")
        return cls.from_path(path)

    @classmethod
    def from_dict(cls: Type[C], config_dict: Dict[str, Any]) -> C:
        """Load configuration from dictionary.

        Ignores all keys not in the class signature.
        """
        valid_args = [str(param) for param in signature(cls).parameters]
        config_dict = {
            arg: config_dict[arg]
            for arg in config_dict if arg in valid_args
        }

        return _build_class(cls, **config_dict)

    @abstractmethod
    def as_dict(self) -> dict:
        """Convert configuration to dictionary."""

    def save(self, checkpoint: Union[Path, str]):
        """Save configuration to checkpoint directory."""
        checkpoint = Path(checkpoint)
        json.dump(self.as_dict(),
                  open(checkpoint.joinpath(f"{self.config_name}.json"), "w"),
                  indent=2)


T = TypeVar("T")  #pylint: disable=invalid-name


def _build_class(cls: Type[T], **kwargs: Any) -> T:
    """Instantiate class with given keyword arguments.

    Ignores all keys not in the class signature.
    """
    valid_args = [str(param) for param in signature(cls).parameters]
    config_dict = {arg: kwargs[arg] for arg in kwargs if arg in valid_args}

    return cls(**config_dict)  # type: ignore


class ModelConfig(_ConfigMixin):
    """Class for configuration of classifier and feature hyperparameters."""

    config_name = "model"
    models = ["LinearClassifier", "BertFinetuning"]
    defaults: Dict[str, Dict[str, Any]] = {
        "LinearClassifier": {
            "use_embedding": True,
            "use_termdocs": True,
            "pretrained_embedding": "fasttext_german",
            "lemmatize": True,
            "ccr": True
        },
        "BertFinetuning": {
            "model_checkpoint": "bert-base-german-dbmdz-uncased"
        }
    }

    def __init__(self,
                 model_name: str = "LinearClassifier",
                 params: Optional[Dict[str, Any]] = None):

        self.model_name = model_name

        if self.model_name not in self.models:
            raise ValueError(f"Name must be in {self.models}")

        if params is None:
            params = {}

        self.fields = params.pop("fields", None)
        self.collection_type = "Unknown"
        self.params: Dict[str, Any] = {**self.defaults[model_name], **params}
        self.checkpoint: Optional[Path] = None
        self._device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

    def as_dict(self):
        return {
            "model_name": self.model_name,
            "fields": self.fields,
            "collection_type": self.collection_type,
            "params": self.params,
        }

    @classmethod
    def from_checkpoint(cls: Type[ModelConfig],
                        checkpoint: Union[Path, str]) -> ModelConfig:
        """Load model config from checkpoint directory.

        Uses saved weight for model building.
        """
        model_config = cls.from_dir(checkpoint)
        model_config.checkpoint = Path(checkpoint)
        return model_config

    def build_model(self, collection: Collection) -> Classifier:
        """Build classifier and features with given config.

        Args:
            descriptions: Document descriptions to use. If None, use
                default descriptions.
            checkpoint: If not None, load model weights from this directory.

        Returns:
            Configured Classifier.
        """
        self.fields = self.fields or collection.fields
        self.collection_type = type(collection).__name__
        logger.info("Building features and model.")

        clf: Classifier
        if self.model_name == "LinearClassifier":
            clf = self._build_linear(collection)
        else:
            clf = self._build_bert(collection)
        clf.to(self._device)
        return clf

    def _build_embedding(self) -> Embedding:

        pretrained = self.params["pretrained_embedding"]
        try:
            checkpoint = validate_checkpoint(pretrained)
            pool = _build_class(BertPool,
                                model_checkpoint=checkpoint,
                                **self.params)
            pool.to(self._device)
            return pool

        except ValueError:
            pass

        we = WordEmbedding.from_path(pretrained)
        tokenizer = Tokenizer(lemmatize=self.params["lemmatize"])
        return _build_class(SumWordsEmbedding,
                            we=we,
                            tokenizer=tokenizer,
                            **self.params)

    def _build_linear(self, collection: Collection) -> LinearClassifier:
        feature_list: List[ScoreFeatures] = []
        if self.params["use_embedding"]:
            emb = self._build_embedding()

            cache: Optional[StringCache] = None
            emb_path = WEIGHTS_DIR.joinpath(
                self.params["pretrained_embedding"])
            if emb_path.exists() and emb_path.joinpath("cache.pt").exists():
                cache = StringCache.from_path(emb_path, device=emb.device)
            cached_embs = _build_class(CachedEmbeddings,
                                       collection=collection,
                                       embedding=emb,
                                       fields=self.fields,
                                       cache=cache,
                                       **self.params)
            emb_simil = EmbSimil(cached_embs)
            feature_list.append(emb_simil)
        if self.params["use_termdocs"]:
            termdocs = _build_class(TermDocs,
                                    collection=collection,
                                    fields=self.fields,
                                    **self.params)
            feature_list.append(termdocs)
        if not feature_list:
            raise ValueError(
                "At least one of [use_embedding, use_termdocs] must be True.")
        features = ConcatFeatures(feature_list)
        clf = LinearClassifier(features=features)

        if self.checkpoint is not None:
            logger.info("Loading weights.")
            clf.load_weights(self.checkpoint.joinpath("model.pt"))

        return clf

    def _build_bert(self, collection: Collection) -> BertFinetuning:
        pool = _build_class(BertPool, **self.params)
        tokenizer = BertTokenizer(self.params["model_checkpoint"])
        features = BertFeatures(collection=collection,
                                tokenizer=tokenizer,
                                fields=self.fields)
        clf = BertFinetuning(features, pool)

        return clf


@dataclass
class TrainConfig(_ConfigMixin):
    """Collection of training hyperparameters."""
    lr: float = 0.01
    weight_decay: float = 0.0
    num_prefilter: int = 50
    queries_per_batch: int = 5
    sample_size: int = 10
    epochs: int = 1

    config_name = "train"

    def as_dict(self):
        return asdict(self)
