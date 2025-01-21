from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from ..documents import Collection, Document


class LeikaDesc(Document):
    """LeiKa description as data class.

    Args:
        code: LeiKa code
        name: LeiKa name
        group: Internal classification
        method: Type of service
        synonyms: Provided list of synonyms for the LeiKa code
        other: Concatenation of other possibly relevant fields
    """

    code: int
    name: str
    group: str
    method: str
    synonyms: List[str]
    other: List[str]

    def __int__(self) -> int:
        return self.code

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> LeikaDesc:
        """Build instance from dictionary.

        Args:
            desc: Description record as dict.

        Returns:
            Parsed description

        Raises:
            KeyError: If one of the required fields was not found.
            ValueError: If code could not be parsed as integer.
        """

        try:
            code_str = d["schluessel"]
            name = d["kennung"]
            group = d["gruppierung"]
            method = " ".join([d["verrichtung"], d["verrichtungsdetail"]])
            other = _split_list(d["besondere_merkmale"])
            synonyms = _split_list(d["synonyme"])
        except KeyError as e:
            raise KeyError(f"Field {str(e)} not found.")
        try:
            code = int(code_str)  #type: ignore
        except ValueError:
            raise ValueError(f"Schluessel {code_str} not an integer.")

        return cls(code=code,
                   name=name,
                   group=group,
                   method=method,
                   synonyms=synonyms,
                   other=other)


def _split_list(s: Union[str, List[str]]) -> List[str]:

    if isinstance(s, str):
        s = s.split("|")
    return [word for word in s if word]


class LeikaDescriptions(Collection[LeikaDesc]):
    @classmethod
    def from_path(cls, desc_path: Union[Path, str]) -> LeikaDescriptions:
        """Load Leika descriptions from json path."""
        raw_descs = json.load(open(desc_path, "r"))
        descs = [LeikaDesc.from_dict(d) for d in raw_descs]
        return cls(descs)

    @property
    def string_fields(self) -> List[str]:
        return ["name", "group", "method"]

    @property
    def list_fields(self) -> List[str]:
        return ["synonyms", "other"]

    def string_entries(self,
                       field: str,
                       codes: Optional[List[int]] = None) -> List[str]:
        if field not in self.string_fields:
            raise ValueError(f"String field must be contained in" +
                             f"{', '.join(self.string_fields)}." +
                             f"Got {field}.")
        if codes is None:
            codes = self.codes
        return [getattr(self.docs[code], field) for code in codes]

    def list_entries(self,
                     field: str,
                     codes: Optional[List[int]] = None) -> List[List[str]]:
        if field not in self.list_fields:
            raise ValueError(f"String field must be contained in" +
                             f"{', '.join(self.list_fields)}." +
                             f"Got {field}.")
        if codes is None:
            codes = self.codes
        return [getattr(self.docs[code], field) for code in codes]


def parse_descs(
    raw_docs: List[Dict[str, Any]]
) -> Tuple[LeikaDescriptions, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Try to parse list of dictionaries into Leika descriptions.

    Args:
        raw_docs: List of dicts to parse.

    Returns:
        Tuple of Leika descriptions, accepted entries and rejected entries.

    Raises:
        ValueError if no valid entries could be found.
    """

    desc_list: List[LeikaDesc] = []
    accepted: List[Dict[str, str]] = []
    rejected: List[Dict[str, str]] = []

    for d in raw_docs:
        try:
            desc = LeikaDesc.from_dict(d)
        except (KeyError, ValueError) as e:
            d["Error"] = str(e)
            rejected.append(d)
        else:
            desc_list.append(desc)
            accepted.append(d)

    if not desc_list:
        raise ValueError("No valid documents found.")

    return LeikaDescriptions(desc_list), accepted, rejected
