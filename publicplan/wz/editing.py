import copy
import logging
from typing import Any, Dict, List, Tuple

from pydantic import BaseModel  # pylint: disable=no-name-in-module
from pydantic import ValidationError

from .description import WZCode, WZDesc, WZDescriptions

logger = logging.getLogger(__name__)


class WZAddition(BaseModel):
    """Additions to WZ descriptions supplied by editing system."""
    code: WZCode
    name: str = ""
    section_name: str = ""
    group_name: str = ""
    class_name: str = ""
    explanation: str = ""
    keywords: List[str] = []


def _join(s1: str, s2: str) -> str:
    return " ".join([s1, s2])


def _update_desc(desc: WZDesc, add: WZAddition) -> WZDesc:

    new_keywords = desc.keywords + [
        kw for kw in add.keywords if kw not in desc.keywords
    ]
    return WZDesc(code=desc.code,
                  name=_join(desc.name, add.name),
                  section_name=_join(desc.section_name, add.section_name),
                  group_name=_join(desc.group_name, add.group_name),
                  class_name=_join(desc.class_name, add.class_name),
                  explanation=_join(desc.explanation, add.explanation),
                  exclusions=desc.exclusions,
                  keywords=new_keywords)


def update_descs(
    descs: WZDescriptions, additions: List[WZAddition]
) -> Tuple[WZDescriptions, Dict[WZCode, WZAddition], List[WZAddition]]:
    """Apply additions from editing systems to descs.

    Args:
        descs: Original descriptions.
        additions: Additions supplied by editing system.

    Returns:
        New descriptions, accepted additions, rejected additions.
    """

    new_descs = {desc.code: copy.deepcopy(desc) for desc in descs.values()}
    accepted: Dict[WZCode, WZAddition] = {}
    rejected: List[WZAddition] = []

    for add in additions:
        code = add.code
        try:
            desc = descs[code]
        except KeyError:
            rejected.append(add)
            logger.warning(
                f"Rejecting WZ addition. Code {code} not recognized.")
            continue
        if code in accepted:
            logger.warning(
                f"Code {code} already updated. Ignoring second entry.")
            continue
        accepted[code] = add
        new_descs[code] = _update_desc(desc, add)

    return WZDescriptions(list(new_descs.values())), accepted, rejected


def parse_additions(
    raw_adds: List[Dict[str, Any]]
) -> Tuple[List[WZAddition], List[Dict[str, Any]]]:
    """Parse additions from json response.

    Args:
        raw_adds: Raw Json response.
    Returns:
        Successfully parsed additions and retrieval results rejected
        during parsing.
    """
    accepted: List[WZAddition] = []
    rejected: List[Dict[str, Any]] = []
    for raw_add in raw_adds:
        try:
            add = WZAddition(**raw_add)
            accepted.append(add)
        except ValidationError as e:
            logger.warning("Rejecting WZ addition.")
            logger.warning(str(e))
            rejected.append(raw_add)

    return accepted, rejected
