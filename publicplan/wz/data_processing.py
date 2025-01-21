from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union, DefaultDict
from zipfile import ZipFile

import numpy as np
import pandas as pd

from publicplan.paths import (WZ_DATA_DIR, WZ_DATA_PATH, WZ_DATA_ZIP,
                              WZ_IHK_PATH, WZ_TEST_PATH, WZ_TRAIN_PATH,
                              WZ_VAL_PATH)

from .description import WZCode

_IHK_ABBREV = {
    "GH": "Großhandel",
    "EH": "Einzelhandel",
    "HV": "Handelsvermittlung",
    "H.v.": "Herstellung von",
    "H. ": "Herstellung ",
    " v. ": " von ",
    " u. ": " und ",
    " f. ": " für ",
    " m. ": " mit ",
    " ä.": " ähnlichem",
    "\xa0": " "
}


def _ihk_code(code: str):
    parts = [code[:2], code[2:4], code[4:5], code[5:]]
    return ".".join(part for part in parts if part)


def _ihk_category(code: str):
    part_lengths = {
        1: "sections",
        2: ["groups", "classes"],
        3: "subclasses",
        4: "additions"
    }
    parts = code.split(".")
    if len(parts) == 2:
        if len(parts[1]) == 1:
            return "groups"
        return "classes"
    return part_lengths[len(parts)]


def process_ihk(
        data: pd.DataFrame,
        expand_abbrev: bool = True) -> Dict[str, List[Tuple[str, str]]]:
    """Extract ihk names from parsed csv file.

    Args:
        data: Parsed csv file.
        expand_abbrev: Whether to expand abbrevations.

    Returns: Dictionary indexed by categories
        (sections, groups, classes, subclasses, additions).
        The corresponding values are pairs of ihk code and name.
    """
    names: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for _, row in data.iterrows():
        code = _ihk_code(row[0])
        name = row[1]
        if expand_abbrev:
            for abbrev, expanded in _IHK_ABBREV.items():
                name = name.replace(abbrev, expanded)
        category = _ihk_category(code)
        names[category].append((code, name))

    return names


def ihk_keywords(
        ihk_data: Optional[pd.DataFrame] = None
) -> DefaultDict[WZCode, List[str]]:
    """Extract keywords from ihk data.

    The IHK hierarchy is flattened to the WZ level
    and the lower subcategories are used as additional keywords.
    """
    if ihk_data is None:
        ihk_data = pd.read_csv(WZ_IHK_PATH, dtype={"Schlüssel": "object"})
    additions = process_ihk(ihk_data)["additions"]
    result: DefaultDict[WZCode, List[str]] = defaultdict(list)
    for code, add in additions:
        code_split = code.split(".")
        wz_code = ".".join(code_split[:3])
        wz_code = WZCode.parse(wz_code)
        result[wz_code].append(add)

    return result


def load_gwa(only_complete: bool = False,
             drop_duplicates: bool = False) -> pd.DataFrame:
    """Load gwa training data.

    Args:
        only_complete: Only include data with complete codes.
        drop_duplicates: Drop entries with the same code and description.

    Returns:
        Dataframe with na entries dropped. The state codes are
        replaced with the corresponding names.

    Raises:
        FileNotFoundError if file was not found in expected location.
    """

    if not WZ_DATA_PATH.exists():
        if WZ_DATA_ZIP.exists():
            with ZipFile(WZ_DATA_ZIP, "r") as zf:
                zf.extractall(WZ_DATA_DIR)
        else:
            raise FileNotFoundError(f"Training data not found.")
    data = pd.read_csv(
        WZ_DATA_PATH,
        header=0,
        delimiter=";",
        dtype={
            "WZ": object,  # Need to distinguish 0dddd from dddd
            "ArtBetrieb": object  # Binary Code for 2^4 possibilities
        })
    data.dropna(inplace=True)

    state_codes = {
        1: "Schleswig-Holstein",
        2: "Hamburg",
        3: "Niedersachsen",
        4: "Bremen",
        5: "Nordrhein-Westfalen",
        6: "Hessen",
        7: "Rheinland-Pfalz",
        8: "Baden-Württemberg",
        9: "Bayern",
        10: "Saarland",
        11: "Berlin",
        12: "Brandenburg",
        13: "Mecklenburg-Vorpommern",
        14: "Sachsen",
        15: "Sachsen-Anhalt",
        16: "Thüringen"
    }

    data["Land"] = data["Land"].map(state_codes)
    data["FullCode"] = data["WZ"].map(_parse_code)

    if only_complete:
        data.dropna(inplace=True)

    if drop_duplicates:
        data.drop_duplicates(["Taetigkeit", "WZ"], inplace=True)

    return data


def _parse_code(code: str) -> Union[WZCode, float]:
    try:
        parsed = WZCode.parse(code)
    except ValueError:
        parsed: float = np.nan  # type: ignore
    return parsed


def _data_by_state(data: pd.DataFrame, state: str) -> pd.DataFrame:
    state_data = data[data["Land"] == state]

    return state_data.loc[:, ["Taetigkeit", "FullCode"]]


def train_val_test_split(
        test_ratio: float = 0.2,
        seed: int = 42,
        save: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Perform train-validation-test split.

    The split is performed evenly for each state.

    Args:
        test_ratio: Proportion of data used for validation and
            test datasets (split evenly between the two).
        seed: Random seed for deterministic splitting.
        save: Whether to save corresponding split

    Returns:
        Train, validation and test datasets.
    """

    data = load_gwa(only_complete=True, drop_duplicates=True)
    states = set(data["Land"].values)
    train_ratio = 1 - test_ratio

    train_list: List[pd.DataFrame] = []
    val_list: List[pd.DataFrame] = []
    test_list: List[pd.DataFrame] = []

    for state in states:
        state_data = data[data["Land"] == state]
        train = state_data.sample(frac=train_ratio, random_state=seed)
        rest = state_data.drop(train.index)
        test = rest.sample(frac=0.5, random_state=seed)
        val = rest.drop(test.index)

        train_list.append(train)
        val_list.append(val)
        test_list.append(test)

    train_data = pd.concat(train_list)
    val_data = pd.concat(val_list)
    test_data = pd.concat(test_list)

    if save:
        train_data.to_csv(WZ_TRAIN_PATH)
        val_data.to_csv(WZ_VAL_PATH)
        test_data.to_csv(WZ_TEST_PATH)

    return train_data, val_data, test_data


if __name__ == "__main__":
    train_val_test_split(save=True)
