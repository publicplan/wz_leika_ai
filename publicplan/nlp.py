import json
from pathlib import Path
from typing import Dict, List, Union

from spacy.lang.de import German
from spacy.tokens import Token

from .paths import IWNLP_PATH


class Tokenizer:
    """Helper class for splitting and filtering texts.

    Args:
        filter_stops: Omit stop words.
        filter_punct: Omit punctutation.
        filter_connecting: Replace connecting symbols like - by spaces.
        lemmatize: Perform simple lemmatization on words.
    """
    def __init__(self,
                 filter_stops: bool = True,
                 filter_punct: bool = True,
                 filter_connecting: bool = True,
                 lemmatize: bool = False):

        self.filter_stops = filter_stops
        self.filter_punct = filter_punct
        self.filter_connecting = filter_connecting
        self._lemmatizer = None
        if lemmatize:
            self._lemmatizer = Lemmatizer()

        self._connecting = ["-", "/", "_"]

        self._base_nlp = German()

    def _filter_token(self, token: Token) -> bool:

        stop = self.filter_stops and token.is_stop
        punct = self.filter_punct and token.is_punct
        space = token.is_space

        return not any((stop, punct, space))

    def split(self, text: str) -> List[str]:
        """Split text and filter and transform resulting tokens."""

        if self.filter_connecting:
            for s in self._connecting:
                text = text.replace(s, " ")

        doc = self._base_nlp(text)
        tokens = list(filter(self._filter_token, doc))

        if self._lemmatizer:
            return [self._lemmatizer.lemma(token.text) for token in tokens]

        return [token.text for token in tokens]


def build_lemma_dict(dict_path: Union[Path, str]) -> Dict[str, List[str]]:
    """Parse lemma dictionary.

    Args:
        dict_path: Path to IWNLP lemma dictionary.

    Returns:
        Dictionary with (lowercased) words as keys and list of lemmas as values.
    """
    lemma_dict: Dict[str, List[str]] = {}
    lemma_list = json.load(open(dict_path, "r"))
    for entry in lemma_list:
        word = entry["Form"].lower()
        lemmas = [l["Lemma"] for l in entry["Lemmas"]]
        lemma_dict[word] = lemmas

    return lemma_dict


class Lemmatizer:
    """Wrapper around iwnlp lemmatizer.

    Args:
        dict_path: Path to lemma dictionary.
    """
    def __init__(self, dict_path: Union[Path, str] = IWNLP_PATH):

        self._lemma_dict = build_lemma_dict(dict_path)

    def lemma(self, word: str) -> str:
        """Lemmatize word.

        Lemma lookup is case-insensitive.
        If multiple lemmas are possible, returns the first one.
        If no lemmas are found, return the original word.
        """

        lemmas = self.lemmas(word)
        if lemmas:
            return lemmas[0]

        return word

    def lemmas(self, word: str) -> List[str]:
        """Return all found lemmas of given word."""

        if word.lower() not in self._lemma_dict.keys():
            return []
        return self._lemma_dict[word.lower()]
