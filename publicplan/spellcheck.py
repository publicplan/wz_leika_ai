import re
from pathlib import Path
from typing import Dict, Optional, Pattern, Set, Union

import pandas as pd
from symspellpy import SymSpell, Verbosity

from .nlp import build_lemma_dict
from .paths import BASE_SPELLDICT_PATH, IWNLP_PATH, SPELLDICT_PATH


class SpellChecker:
    """Spell check wrapper around symspell.

    Corrects words by mapping misspelled word to dictionary word with
    shortest edit distance and breaks ties through frequency count.

    Args:
        dictionary: Path to frequency dictionary.
        edit_distance: Maximal edit distance to consider for correction.
    """
    def __init__(self, dictionary: Union[Path, str], edit_distance: int = 2):

        self.edit_distance = edit_distance
        self._checker = SymSpell(max_dictionary_edit_distance=edit_distance)
        self._checker.load_dictionary(dictionary, 0, 1)

        self._split_re: Pattern = re.compile(r"([\W])")
        self._ignore_res = [re.compile(r"\d"), re.compile(r"[A-Z]{2,}")]
        self._ignore_res.append(self._split_re)

    def correct(self, s: str) -> str:
        """Correct string."""
        tokens = self._split_re.split(s)
        corrected = [self._correct_token(token) for token in tokens if token]

        return "".join(corrected)

    def _correct_token(self, token: str) -> str:

        if any(r.match(token) for r in self._ignore_res):
            return token

        suggestions = self._suggestions(token.lower())
        if suggestions:
            suggestion: str = suggestions[0].term
            if token[0].isupper():
                return suggestion.capitalize()
            return suggestion
        return token

    def _suggestions(self, word: str):
        return self._checker.lookup(word,
                                    Verbosity.TOP,
                                    max_edit_distance=self.edit_distance)

    def add_words(self, words: Set[str], count: int = 1000) -> Set[str]:
        """Add words to spelling dictionary.

        Args:
            words: List of words to add.
            count: Artificial count to give new words.

        Returns:
            Added words.
        """
        added: Set[str] = set()
        for word in words:
            tokens = self._split_re.split(word)
            for token in tokens:
                if len(token) <= 1:
                    continue
                if any(r.match(token) for r in self._ignore_res):
                    continue
                token = token.lower()
                self._checker.create_dictionary_entry(token, count)
                added.add(token)

        return added


def add_lemmas(word_freq: Dict[str, int],
               dict_path: Union[Path, str] = IWNLP_PATH,
               discount: float = 0.5,
               exclude_re: Optional[Pattern] = None) -> None:
    """Add words from lemmatization dictionary to word frequency dictionary.

    Args:
        word_freq: Frequency dictionary to update.
        dict_path: Path to lemmatization dictionary.
        discount: Discount factor to apply to word counts.
    """
    lemma_dict = build_lemma_dict(dict_path)
    for word in lemma_dict:
        if (exclude_re and exclude_re.search(word)) or word in word_freq:
            continue
        count = 1
        for lemma in lemma_dict[word]:
            lemma = lemma.lower()
            if lemma in word_freq:
                count = max(count, int(word_freq[lemma] * discount))
        word_freq[word] = count


def _create_wordfreq_file(exclude_re: Pattern, threshold: int) -> None:

    word_df = pd.read_csv(BASE_SPELLDICT_PATH,
                          header=None,
                          index_col=0,
                          sep=" ",
                          skiprows=4,
                          names=["count", "word"],
                          keep_default_na=False)

    word_df["count"] = (word_df["count"] * 100).astype(int)
    word_df["word"] = word_df["word"].apply(lambda w: w.lower())

    def filter_word(word: str, exlude_re: Pattern) -> bool:
        if len(word) <= 1:
            return False
        if exlude_re.search(word) is not None:
            return False
        return True

    included = word_df[word_df["word"].apply(filter_word,
                                             exlude_re=exclude_re)]
    word_freq = included.set_index("word").to_dict()["count"]
    add_lemmas(word_freq, IWNLP_PATH, exclude_re=exclude_re)

    with open(SPELLDICT_PATH, "w") as out_file:
        for word, freq in word_freq.items():
            if freq > threshold:
                out_file.write(f"{word} {freq}\n")


if __name__ == "__main__":
    _create_wordfreq_file(exclude_re=re.compile("[^a-zA-ZÄäöÖüÜß]"),
                          threshold=0)
