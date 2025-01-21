from __future__ import annotations

from abc import abstractmethod
from typing import (Any, Dict, Iterator, List, Mapping, Optional, Sequence,
                    TypeVar)

from pydantic import BaseModel  # pylint: disable=no-name-in-module


class Document(BaseModel):
    @abstractmethod
    def __int__(self) -> int:
        pass


DocType = TypeVar("DocType", bound=Document)


class Collection(Mapping[int, DocType]):
    """Class for collection of documents.

    Args:
        docs: List of Documents.

    Raises:
        ValueError if collection is empty or contains duplicate codes.
    """
    def __init__(self, docs: Sequence[DocType]):

        if not docs:
            raise ValueError("Document collection is empty.")
        self.codes = [int(d) for d in docs]
        if len(set(self.codes)) != len(docs):
            raise ValueError("Collection contains duplicate codes.")
        self.docs = {int(d): d for d in docs}
        self.inverted_index = {code: i for i, code in enumerate(self.codes)}

    def __getitem__(self, key: int) -> DocType:
        return self.docs[key]

    def __len__(self) -> int:
        return len(self.codes)

    def __iter__(self) -> Iterator[int]:
        yield from self.docs

    def dict(self) -> Dict[int, Dict[str, Any]]:
        return {code: self.docs[code].dict() for code in self.codes}

    @property
    def fields(self) -> List[str]:
        """List of available fields."""
        return self.string_fields + self.list_fields

    @property
    @abstractmethod
    def string_fields(self) -> List[str]:
        """List of fields consisting of strings."""

    @property
    @abstractmethod
    def list_fields(self) -> List[str]:
        """List of fields consisting of List of strings."""

    @abstractmethod
    def string_entries(self,
                       field: str,
                       codes: Optional[List[int]] = None) -> List[str]:
        """
        Get entries corresponding to string field for all documents in collection.

        Args:
            field: Field to return entries from.
            codes: If not None, restrict result to these codes.

        Return:
            List of field entries.
        """

    @abstractmethod
    def list_entries(self,
                     field: str,
                     codes: Optional[List[int]] = None) -> List[List[str]]:
        """
        Get entries corresponding to list field for all documents in collection.

        Args:
            field: Field to return entries from.
            codes: If not None, restrict result to these codes.

        Return:
            List of field entries.
        """
