from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, DefaultDict

from ..documents import Collection, Document
from ..paths import WZ_DESCS_PATH, WZ_KEYWORDS_PATH

logger = logging.getLogger(__name__)

ANG_CODES = [
    "01.19.9", "08.99.0", "10.89.0", "13.99.0", "14.19.0", "16.29.0",
    "18.12.0", "20.59.0", "23.69.0", "23.70.0", "23.99.0", "25.62.0",
    "25.73.3", "25.99.3", "27.90.0", "28.13.0", "28.14.0", "28.29.0",
    "28.49.9", "28.99.0", "30.99.0", "31.09.9", "32.50.1", "32.99.0",
    "33.17.0", "33.20.0", "42.99.0", "43.29.9", "43.39.0", "43.99.9",
    "46.14.1", "46.14.9", "46.15.2", "46.15.4", "46.18.9", "46.38.9",
    "46.49.5", "46.69.1", "47.52.1", "47.59.9", "47.78.9", "47.99.9",
    "49.39.9", "52.21.9", "52.22.9", "52.23.9", "52.29.9", "55.90.9",
    "61.90.9", "63.99.0", "64.99.9", "69.10.9", "74.90.0", "77.39.0",
    "81.29.9", "82.99.9", "85.59.9", "88.99.0", "93.29.0", "94.99.9", "96.09.0"
]


class WZCode(str):
    """Wrapper class for WZ code.

    A (full) WZ Code has the form 'dd.dd.d' where 'd' denotes a digit.

    The input must have either this format or be a string consisting of
    exactly 5 digits.
    """
    @classmethod
    def __get_validators__(cls):
        yield cls.parse

    @classmethod
    def parse(cls, v: Any) -> WZCode:
        """Parse string as WZ Code.

        Args:
            v: String to parse.

        Returns:
            Parsed code in WZ format.

        Raises:
            TypeError, if input is not a string.
            ValueError, if code could not be parsed.
        """

        if not isinstance(v, str):
            raise TypeError("WZCode must be str.")

        parts = [int(p) for p in v.split(".")]
        if not len(parts) in [1, 3]:
            raise ValueError(f"Can not parse {v} as WZCode.")

        if len(parts) == 1:
            if len(v) == 5:
                code = parts[0]
                parts = [code // 1000, (code % 1000) // 10, code % 10]
            else:
                raise ValueError(f"Integer Code {v} is ambigous as WZCode.")

        parsed = f"{parts[0]:02}.{parts[1]:02}.{parts[2]}"

        return cls(parsed)

    def __int__(self):
        parts = self.split(".")
        return sum(int(part) * d for part, d in zip(parts, [1000, 10, 1]))


class WZDesc(Document):
    """WZ description extracted from destatis data.

    Attr:
        code: WZ code, consisting of section, class and subclass codes.
        name: Name of WZ classification.
        section_name: Section of corresponding section.
        group_name: Name of corresponding group.
        class_name: Name of corresponding class.
        explanation: Explanation of WZ code.
        exclusions: Excluded entities from domain of WZ code.
    """

    code: WZCode
    name: str
    section_name: str
    group_name: str
    class_name: str
    explanation: str
    exclusions: str
    keywords: List[str]

    def __int__(self) -> int:
        return int(self.code)


class WZDescriptions(Collection[WZDesc]):
    def __init__(self, docs: Sequence[WZDesc]):
        super().__init__(docs)
        self.deletions = [re.compile(r) for r in [r"\([oO]hne [^)]*\)"]]

    def __getitem__(self, key: Union[str, int]) -> WZDesc:
        if isinstance(key, str):
            key = int(WZCode.parse(key))
            return self.docs[int(key)]

        return self.docs[int(key)]

    # Ignore exclusions
    @property
    def string_fields(self) -> List[str]:
        return [
            "name",
            "section_name",
            "group_name",
            "class_name",
            "explanation",
        ]

    @property
    def list_fields(self) -> List[str]:
        return ["keywords"]

    # Make this less ugly
    def string_entries(self,
                       field: str,
                       codes: Optional[List[int]] = None) -> List[str]:
        if field not in self.string_fields:
            raise ValueError(f"String field must be contained in" +
                             f"{', '.join(self.string_fields)}." +
                             f"Got {field}.")
        if codes is None:
            codes = self.codes
        return [
            self._apply_deletions(getattr(self.docs[code], field))
            for code in codes
        ]

    def list_entries(self,
                     field: str,
                     codes: Optional[List[int]] = None) -> List[List[str]]:

        if field not in self.list_fields:
            raise ValueError(f"List field must be contained in" +
                             f"{', '.join(self.list_fields)}." +
                             f"Got {field}.")
        if codes is None:
            codes = self.codes

        return [[
            self._apply_deletions(s) for s in getattr(self.docs[code], field)
        ] for code in codes]

    def _apply_deletions(self, s: str) -> str:
        for r in self.deletions:
            s = r.sub("", s)

        return s


def build_descriptions(
        descs_path: Union[Path, str] = WZ_DESCS_PATH,
        keywords_path: Union[Path, str] = WZ_KEYWORDS_PATH) -> WZDescriptions:
    """Parse description and keywords information from XML files.

    Args:
        descs_path: Path to destatis WZ classification with explanations.
        descs_path: Path to destatis WZ keyword list.

    Returns:
        Dictionary of WZ codes with corresponding description.
    """

    sections, groups, classes, code_items = _parse_items(descs_path)
    code_keywords = _parse_keywords(keywords_path)

    descs: List[WZDesc] = []

    for code, item in code_items.items():
        try:
            section_name = sections[code[0]]
            group_name = groups[(code[0], code[1] // 10)]
            class_name = classes[code[:2]]
        except KeyError as e:
            logger.error(f"Entry for subcode {e} of code {code} not found.")
            logger.error("Skipping entry.")
            continue

        code_string = f"{code[0]}.{code[1]}.{code[2]}"

        desc = WZDesc(code=code_string,
                      name=item["name"],
                      section_name=section_name,
                      group_name=group_name,
                      class_name=class_name,
                      explanation=item["explanation"],
                      exclusions=item["exclusions"],
                      keywords=code_keywords[code])

        descs.append(desc)

    return WZDescriptions(descs)


def _parse_items(
        descs_path: Union[Path, str]) -> Tuple[dict, dict, dict, dict]:
    root = ET.parse(open(descs_path)).getroot()

    sections: Dict[int, str] = {}
    groups: Dict[Tuple[int, int], str] = {}
    classes: Dict[Tuple[int, int], str] = {}
    code_items: Dict[Tuple[int, int, int], Dict[str, str]] = {}

    for item in root.iter('Item'):

        level = item.attrib['idLevel']

        name_node = item.find(".//LabelText[@language='DE']")
        if name_node is not None and name_node.text is not None:
            name = name_node.text
        else:
            logger.error("No name found for item. Skipping item.")
            continue

        # Ignore highest level in the hierarchy
        if level == "1":
            continue

        code = _get_code(item)

        code_lengths = {"2": 1, "3": 2, "4": 2, "5": 3}
        try:
            expected = code_lengths[level]
            assert expected == len(code)
        except AssertionError:
            logger.error(
                f"Level mismatch: Expected code to have length {expected}." +
                f"Got {len(code)}. Skipping entry.")
            continue
        # Section level
        if level == "2":
            sections[code[0]] = name
        # Group level
        elif level == "3":
            groups[code] = name  # type: ignore
        # Class level
        elif level == "4":
            classes[code] = name  # type: ignore
        # Subclass level
        elif level == "5":
            explanation = ""
            exclusions = ""
            prop = item.find("Property[@name='ExplanatoryNote']")
            if prop is not None:

                def xpath(name: str) -> str:
                    path = "./PropertyQualifier"  # prop type
                    path += f"[@name='{name}']"  # filter name attr.
                    path += "[@language='DE']"  # filter language attr.
                    path += "/PropertyText"
                    return path

                explanation = _get_content(prop, xpath("CentralContent"))
                exclusions = _get_content(prop, xpath("Exclusions"))

                # Translate unicode bullet symbol
                explanation = explanation.replace("\u25CF", "*")
                explanation = explanation.replace("\u25CF", "*")

            code_items[code] = {  # type: ignore
                "name": name,
                "explanation": explanation,
                "exclusions": exclusions
            }

    return sections, groups, classes, code_items


def _get_code(item: ET.Element) -> Tuple[int, ...]:
    code = tuple(int(part) for part in item.attrib['id'].split("."))
    return code


def _get_content(prop: ET.Element, xpath: str) -> str:
    content = prop.find(xpath)
    if content is not None and content.text is not None:
        return content.text

    return ""


def _parse_keywords(
        keywords_path: Union[Path, str]) -> Dict[Tuple[int, ...], List[str]]:
    code_keywords: Dict[Tuple[int, ...], List[str]] = defaultdict(list)

    root = ET.parse(open(keywords_path)).getroot()

    keyword_path = "./PropertyQualifier[@language='DE']"
    keyword_path += "/PropertyText[@type='Content']"
    for item in root.findall(".//Item[@idLevel='5']"):
        code = _get_code(item)  # type: ignore
        keywords: List[str] = []

        for prop in item.findall("./Property[@name='Keyword']"):
            content = _get_content(prop, keyword_path)
            if content:
                keywords.append(content)

        code_keywords[code] = keywords

    return code_keywords


def build_gp2019a(
        keywords_path: Union[Path, str]) -> DefaultDict[WZCode, List[str]]:
    code_keywords: DefaultDict[WZCode, List[str]] = defaultdict(list)

    root = ET.parse(open(keywords_path)).getroot()

    keyword_path = "./PropertyQualifier[@language='DE']"
    keyword_path += "/PropertyText[@type='Description']"
    for item in root.findall(".//Item[@idLevel='5']"):
        code = WZCode(item.attrib["id"])  # type: ignore

        keywords: List[str] = []

        for prop in item.findall("./Property[@genericName='gp2019a']"):
            content = _get_content(prop, keyword_path)
            if content:
                keywords.append(content)

        code_keywords[code] = keywords

    return code_keywords
