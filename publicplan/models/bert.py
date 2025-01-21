import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import transformers as tr
from transformers.configuration_bert import BERT_PRETRAINED_CONFIG_ARCHIVE_MAP
from transformers.optimization import AdamW

from publicplan.documents import Collection
from publicplan.paths import WEIGHTS_DIR

from .classifiers import Classifier
from .embeddings import CachedEmbeddings, Embedding, EmbSimil, StringCache
from .features import Features
from .utils import pad_tensors

logger = logging.getLogger(__name__)

BERT_PRETRAINED_MODELS = list(BERT_PRETRAINED_CONFIG_ARCHIVE_MAP.keys())


def validate_checkpoint(
        model_checkpoint: Union[str, Path]) -> Union[str, Path]:
    """Check if checkpoint is in hugging face models or available locally.

    Args:
        model_checkpoint: Name or path to model config and weights.
    Returns:
        Huggingface name or checkpoint path.
    Raises:
        ValueError if no valid checkpoint could be found.
    """

    if model_checkpoint in BERT_PRETRAINED_MODELS:
        return model_checkpoint
    if Path(model_checkpoint).exists():
        return Path(model_checkpoint)
    model_checkpoint = WEIGHTS_DIR.joinpath(str(model_checkpoint))
    if model_checkpoint.exists():
        return Path(model_checkpoint)

    raise ValueError(f"Checkpoint {model_checkpoint} not found.")


class BertTokenizer:
    """Wrapper class for transformers.BertTokenizer.

    Args:
        model_checkpoint: Pretrained model to load tokenizer from.
    """
    def __init__(self, model_checkpoint: Union[str, Path]):
        self._tokenizer = self._load_tokenizer(model_checkpoint)
        self._max_len = self._tokenizer.max_len_single_sentence

    # pylint: disable=no-self-use
    def _load_tokenizer(self, model_checkpoint: Union[str, Path]):
        model_checkpoint = validate_checkpoint(model_checkpoint)
        return tr.BertTokenizer.from_pretrained(str(model_checkpoint))

    def to_ids(self, text: str) -> torch.Tensor:
        """Tokenize string and map tokens to ids."""

        tokens = self._tokenizer.tokenize(text)
        ids = self._tokenizer.convert_tokens_to_ids(tokens)[:self._max_len]
        full_ids = self._tokenizer.build_inputs_with_special_tokens(ids)

        # pylint: disable=not-callable
        return torch.tensor(full_ids, dtype=torch.long)

    def stack_ids(self, texts: List[str]) -> torch.Tensor:
        """Tokenize texts and stack them along dimension zero."""
        if not texts:
            return self.to_ids("").unsqueeze(0)

        ids = pad_tensors([self.to_ids(text) for text in texts], dims=[0])
        return torch.stack(ids)

    def save_checkpoint(self, checkpoint: Union[Path, str]) -> None:
        self._tokenizer.save_vocabulary(checkpoint)


class BertFeatures(Features):
    """Feature class supplying token ids for use in Bert finetuning.

    Args:
        collection: Document collection.
        tokenizer: Tokenizer to use feature generation.
        fields: Document fields to use. If None, use all available fields.
        max_list_length: Maximal length of list field entries provided by feature.
            Length is reduced by random sampling.
    """
    def __init__(self,
                 collection: Collection,
                 tokenizer: BertTokenizer,
                 fields: Optional[List[str]] = None,
                 max_list_length: int = 20):

        self.collection = collection
        if fields is None:
            fields = collection.fields
        self.fields = fields
        self.list_fields_pos = [
            i for i, field in enumerate(fields)
            if field in collection.list_fields
        ]
        self._tokenizer = tokenizer
        self._codes = collection.codes
        self.max_list_length = max_list_length

    @property
    def codes(self) -> List[int]:
        return self._codes

    def _doc_ids(self, codes: List[int]) -> Tuple[torch.Tensor, ...]:
        doc_ids: List[torch.Tensor] = []
        for field in self.fields:
            if field in self.collection.string_fields:
                ids = self._tokenizer.stack_ids(
                    self.collection.string_entries(field, codes))
                doc_ids.append(ids.unsqueeze(dim=1))
            else:
                entries = self.collection.list_entries(field, codes)
                list_ids: List[torch.Tensor] = []
                for entry in entries:
                    ids = self._tokenizer.stack_ids(entry)
                    list_ids.append(ids)

                list_ids = pad_tensors(list_ids, dims=[0, 1])
                doc_ids.append(torch.stack(list_ids))

        return tuple(doc_ids)

    def get(self, query: str, codes: List[int]) -> Tuple[torch.Tensor, ...]:

        doc_ids = self._doc_ids(codes)
        query_ids = self._tokenizer.to_ids(query).unsqueeze(0)
        query_ids = query_ids.expand((len(codes), -1))

        return (query_ids, *doc_ids)

    def collate_fn(
            self, batches: List[Tuple[torch.Tensor,
                                      ...]]) -> Tuple[torch.Tensor, ...]:
        query_batch = _pad_concat([batch[0] for batch in batches], dims=[1])
        doc_batches = [
            _pad_concat([batch[i + 1] for batch in batches], dims=[1, 2])
            for i in range(len(self.fields))
        ]

        return (query_batch, *doc_batches)


def _pad_concat(tensors: List[torch.Tensor], dims: List[int]) -> torch.Tensor:
    tensors = pad_tensors(tensors, dims=dims)
    return torch.cat(tensors)


def _pad_stack(tensors: List[torch.Tensor], dims: List[int]) -> torch.Tensor:
    tensors = pad_tensors(tensors, dims=dims)
    return torch.stack(tensors)


class BertPool(nn.Module, Embedding):
    """Text embedding based on pooling Bert embeddings.

    Args:
        model_checkpoint: Pretrained model configuration to use as base.
            If this is a string, try to load a pretrained model from huggingface or
                from the embeddings directory.
            If it is a path, it constructs a model from the respective config file
            and tries to load the weights if available.
        pooling: Pooling strategy to use. One of 'cls', 'mean' or 'max'.
        num_bert_layers: Number of Bert transformer layers to use.
        freeze_embeddings: Whether  to freeze wordpiece embeddings in training.
        freezed_layers:  Number of Bert layers to freeze in training.
    """

    pooling_strategies = ["cls", "mean", "max"]

    def __init__(
            self,
            model_checkpoint: Union[str,
                                    Path] = "bert-base-german-dbmdz-uncased",
            pooling: str = "cls",
            num_bert_layers: int = 12,
            freeze_embeddings: bool = True,
            freezed_layers: int = 0):

        super().__init__()

        if pooling not in self.pooling_strategies:
            raise ValueError(
                f"Argument pooling must be in {self.pooling_strategies}.")
        self.pooling = pooling
        self.model_checkpoint = model_checkpoint
        self.num_bert_layers = num_bert_layers
        self._bert, self._tokenizer = self._load_bert(model_checkpoint)

        if freeze_embeddings:
            for param in self._bert.embeddings.parameters():
                param.requires_grad = False
        if freezed_layers > 0:
            for layer in self._bert.encoder.layer[:freezed_layers]:
                for param in layer.parameters():
                    param.requires_grad = False

    def _load_bert(
        self,
        model_checkpoint: Union[Path,
                                str]) -> Tuple[tr.BertModel, BertTokenizer]:

        model_checkpoint = validate_checkpoint(model_checkpoint)
        tokenizer = BertTokenizer(model_checkpoint)

        has_weights = model_checkpoint in BERT_PRETRAINED_MODELS
        has_weights |= Path(model_checkpoint).joinpath(
            tr.WEIGHTS_NAME).exists()
        if has_weights:
            bert = tr.BertModel.from_pretrained(
                str(model_checkpoint), num_hidden_layers=self.num_bert_layers)
        else:
            config = tr.BertConfig.from_pretrained(
                str(model_checkpoint), num_hidden_layers=self.num_bert_layers)
            bert = tr.BertModel(config)

        return bert, tokenizer

    # pylint: disable=arguments-differ
    def forward(self, ids: torch.Tensor) -> torch.Tensor:  # type: ignore
        # Flag the padded values in attention mask
        attention_mask = (ids > 0).float()

        embs = self._bert(ids, attention_mask=attention_mask)[0]
        if self.pooling == "mean":
            pooled = embs.sum(dim=1)
            attention_mask = attention_mask.unsqueeze(2).expand_as(embs)
            pooled /= attention_mask.sum(dim=1).clamp(min=1)
        elif self.pooling == "max":
            pooled, _ = embs.max(dim=1)
        else:
            # Use classifier token
            pooled = embs[:, 0, :]

        return pooled

    def vectorize(self, texts: List[str]) -> torch.Tensor:
        """Pool Bert embeddings for each text.

        Stack them along dimension zero.
        """
        if not texts:
            return torch.zeros((1, self.dim),
                               dtype=torch.float32,
                               device=self.device)
        self.eval()
        chunk_size = 10
        embs: List[torch.Tensor] = []
        for n in range(0, len(texts), chunk_size):
            chunk = texts[n:n + chunk_size]
            ids = self._tokenizer.stack_ids(chunk).to(self.device)
            with torch.no_grad():
                embs.append(self(ids))

        result = torch.cat(embs)
        return result

    @property
    def dim(self) -> int:
        return self._bert.config.hidden_size

    @property
    def device(self) -> torch.device:
        """Device model is loaded on."""
        return next(self.parameters()).device

    def save_checkpoint(self, checkpoint: Union[Path, str]) -> None:
        checkpoint = Path(checkpoint)
        weights_path = checkpoint.joinpath(tr.WEIGHTS_NAME)
        config_path = checkpoint.joinpath(tr.CONFIG_NAME)
        self._bert.config.to_json_file(config_path)

        torch.save(self._bert.state_dict(), weights_path)
        self._tokenizer.save_checkpoint(checkpoint)


class BertFinetuning(Classifier):
    """Relevance classifier based on bert model.

    Intended to finetune underlying bert pooling model.

    Args:
        features: Wordpiece features and optionally Term-Document features.
        pool: Bert model to use for pooling.
    """
    def __init__(self, features: BertFeatures, pool: BertPool):

        super().__init__()

        self._features = features
        self._collection = features.collection
        self.pool = pool
        self._num_fields = len(self._features.fields)

        self.cosine = nn.CosineSimilarity(dim=1)
        self.linear = nn.Linear(self._num_fields, 1)

        nn.init.constant_(self.linear.weight, 1.)

        self._max_list_entries = 20
        self._finetuning_lr = 1e-5
        self._cached_features: Optional[EmbSimil] = None

    @property
    def features(self) -> BertFeatures:
        return self._features

    @property
    def cached_features(self) -> EmbSimil:
        """Similarity features based on cached embeddings."""
        if self._cached_features is None:
            self.update_cache()
        assert self._cached_features is not None
        return self._cached_features

    def update_cache(self) -> None:
        time_start = datetime.now()
        logger.info("Updating cache.")

        string_cache = StringCache()
        string_cache.update_collection(self._features.collection,
                                       embedding=self.pool,
                                       fields=self._features.fields)
        cached_embs = CachedEmbeddings(self._features.collection,
                                       embedding=self.pool,
                                       fields=self._features.fields,
                                       cache=string_cache,
                                       ccr=False)
        self._cached_features = EmbSimil(cached_embs)
        time_delta = datetime.now() - time_start
        minutes, seconds = divmod(round(time_delta.total_seconds()), 60)
        logger.info(f"Took {minutes} minutes {seconds} seconds.")

    # pylint: disable=arguments-differ
    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:  # type: ignore

        query_ids, *doc_ids = inputs
        doc_embs = tuple(self._field_embs(field_ids) for field_ids in doc_ids)
        query_embs = self.pool(query_ids)

        scores = self._compute_scores(query_embs, doc_embs)
        out = self.linear(scores).sigmoid()

        return out

    def _field_embs(self, field_ids: torch.Tensor) -> torch.Tensor:

        size = field_ids.size()

        # Reduce list entries by sampling to fit model on GPU
        list_size = size[1]
        sample_size = min(list_size, self._max_list_entries)
        sampled = sorted(random.sample(range(list_size), k=sample_size))
        field_ids = field_ids[:, sampled, :]

        flattened_ids = field_ids.reshape(size[0] * sample_size, size[2])
        flattened_embs = self.pool(flattened_ids)
        field_embs = flattened_embs.reshape(size[0], sample_size, -1).mean(1)

        return field_embs

    def _compute_scores(self, query_embs: torch.Tensor,
                        doc_embs: Tuple[torch.Tensor, ...]) -> torch.Tensor:

        scores = [
            self.cosine(query_embs.expand_as(field_embs), field_embs)
            for field_embs in doc_embs
        ]

        return torch.stack(scores, dim=1)

    def batch_predict(self,
                      queries: List[ſtr],
                      num_returned: Optional[int] = None) -> List[pd.Series]:

        codes = self.features.codes
        self.eval()
        with torch.no_grad():
            scores = self.cached_features.batch_get(queries, codes)[0]
            preds = self.linear(scores).sigmoid()
            preds = preds.squeeze().cpu().numpy()

        num_codes = len(codes)
        preds_split = [
            preds[i * num_codes:(i + 1) * num_codes]
            for i in range(len(queries))
        ]
        if num_returned is None:
            num_returned = num_codes
        predictions = [
            pd.Series(pred,
                      index=codes).sort_values(ascending=False)[:num_returned]
            for pred in preds_split
        ]

        return predictions

    def optimizer(
            self,
            lr: float,
            weight_decay: float = 0.0) -> optim.Optimizer:  # type: ignore

        set_params = lambda params, lr, weight_decay: {
            "params": params,
            "lr": lr,
            "weight_decay": weight_decay
        }
        param_dicts = [set_params(self.linear.parameters(), lr, weight_decay)]
        param_dicts += [
            set_params(self.pool.parameters(), self._finetuning_lr,
                       weight_decay)
        ]
        return AdamW(param_dicts, lr=lr, weight_decay=weight_decay)

    def describe_weights(self) -> Dict[str, float]:
        description = {
            f"Similarity weight for {field}": self.linear.weight[0, i].item()
            for i, field in enumerate(self.features.fields)
        }
        description["Bias"] = self.linear.bias.item()

        return description

    def load_weights(self, weights_path: Union[Path, str]) -> None:
        """Load weights from path."""
        self.load_state_dict(torch.load(weights_path,
                                        map_location=self.device))

    def save_checkpoint(self, checkpoint: Union[Path, str]) -> None:
        self.pool.save_checkpoint(checkpoint)
        weights_path = Path(checkpoint).joinpath("model.pt")
        torch.save(self.state_dict(), weights_path)
