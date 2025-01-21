import json
import logging
import sys
from pathlib import Path
from typing import List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

from ..paths import (LEIKA_CLEANED_DESCS_PATH, LEIKA_DATA_DIR, LEIKA_DATA_PATH,
                     LEIKA_DESCS_PATH)
from .description import parse_descs

logger = logging.getLogger(__name__)


def process_data(
        data_path: Union[str, Path] = LEIKA_DATA_PATH) -> pd.DataFrame:
    """Process raw data.

    Args:
        data_path: Path to data csv.

    Returns:
        Processed data frame.
    """

    # The file contains a variable number of columns. To read it into pandas,
    # we first have to find the maximum of these.
    with open(data_path, "r") as f:
        max_cols = max(len(line.split(",")) for line in f)

    columns = ["query"] + [("lk" + str(i)) for i in range(1, max_cols)]
    df = pd.read_csv(data_path, names=columns)

    return df


def train_test_val_split(
    data: Optional[pd.DataFrame] = None,
    train_ratio: float = 0.8,
    num_reserved_codes: int = 20,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Construct training, validation and test data sets.

    Also reserve some LeiKa codes for later testing. These will be split from the data
    sets and returned in a separate DataFrame.

    Args:
        data: (Unflattened) DataFrame to split. If None, use processed default dataset.
        train_ratio: Proportion of data to use for training.
        num_reserved_codes: Number of codes to reserve.
        seed: Random seed for deterministic splitting.

    Returns:
        train, validation, test, and reserved datasets
    """

    if data is None:
        data = process_data()

    unique_codes: Set[int] = set()

    code_columns = data.columns[1:]
    for c in code_columns:
        unique_codes.update(data[c].dropna())

    np.random.seed(seed)
    reserved_codes = np.random.choice(list(unique_codes),
                                      size=num_reserved_codes,
                                      replace=False)
    res = pd.concat(data[data[c].isin(reserved_codes)] for c in code_columns)
    data = data.drop(res.index)

    train = data.sample(frac=train_ratio, random_state=seed)
    rest = data.drop(train.index)
    test = rest.sample(frac=0.5, random_state=seed)
    val = rest.drop(test.index)

    return train, test, val, res


def save_as_json(data: pd.DataFrame,
                 out_dir: Union[Path, str],
                 file_name: str,
                 admissible_codes: List[int],
                 overwrite: bool = False) -> Set[int]:
    """Save query data as json.

    Ensures that multiple annotations are handled correctly.

    Args:
        data: Dataframe consisting of queries and (multiple) labels.
        out_dir: Directory to save file to.
        file_name: Name of file.
        admissible_codes: Drop codes which are not in this list.
        overwrite: If true, overwrite file if path exists.

    Returns:
        Set of dropped codes.
    """

    dropped: Set[int] = set()
    result: List[dict] = []

    out_path = Path(out_dir).joinpath(file_name)
    if out_path.exists() and not overwrite:
        logger.info(f"File {out_path} already exists. Skipping.")
        return dropped

    for _, row in data.iterrows():
        codes = [int(code) for code in row[1:].dropna().unique()]
        admitted = [code for code in codes if code in admissible_codes]
        dropped.update(set(codes) - set(admitted))
        if admitted:
            result.append({"query": row[0], "codes": admitted})

    logger.info(f"Writing file {out_path}")
    with open(out_path, "w") as of:
        json.dump(result, of, indent=4)
    logger.info(f"Dropped {len(dropped)} codes.")

    return dropped


def clean_description(json_path: Union[Path, str],
                      out_path: Union[Path, str],
                      error_path: Optional[Union[Path, str]] = None) -> int:
    """Clean LeiKa description data.

    Removes entries which can not be parsed as LeikaDesc classes, and saves
    the result as a new json file.

    Args:
        json_path: Path to Leika description in json format.
        out_path: Path of cleaned description json.
        error_path: Path to logfile for errors and removed entries.
            If this is None, no logfile is written.

    Returns:
        Number of deleted entries
    """

    desc = json.load(open(json_path, "r"))
    _, accepted_desc, rejected_desc = parse_descs(desc)
    errors = len(rejected_desc)

    if error_path is not None:
        with open(error_path, "w") as ef:
            ef.write(f"Number of broken entries: {errors}\n")
            json.dump(rejected_desc, ef, indent=4)

    with open(out_path, "w") as of:
        json.dump(accepted_desc, of, indent=4)

    return errors


def _clean_json():

    error_path = LEIKA_DATA_DIR.joinpath("errors.txt")
    errors = clean_description(LEIKA_DESCS_PATH,
                               LEIKA_CLEANED_DESCS_PATH,
                               error_path=error_path)
    logger.info(f"Writing cleaned json to {LEIKA_CLEANED_DESCS_PATH}.")
    logger.info(f"Fixed {errors} errors.")


def _split():

    if not LEIKA_DATA_PATH.exists():
        logger.error(f"Data file {LEIKA_DATA_PATH} not found")
        sys.exit(1)

    descs = json.load(open(LEIKA_CLEANED_DESCS_PATH, "r"))
    admissible_codes = [int(desc["schluessel"]) for desc in descs]
    processed_data = process_data(LEIKA_DATA_PATH)
    logger.info(f"Writing datasets to {LEIKA_DATA_DIR}.")
    train, val, test, res = train_test_val_split(processed_data)

    dropped: Set[int] = set()
    dropped.update(
        save_as_json(train, LEIKA_DATA_DIR, "train.json", admissible_codes))
    dropped.update(
        save_as_json(val, LEIKA_DATA_DIR, "validation.json", admissible_codes))
    dropped.update(
        save_as_json(test, LEIKA_DATA_DIR, "test.json", admissible_codes))
    dropped.update(
        save_as_json(res, LEIKA_DATA_DIR, "reserved.json", admissible_codes))

    if dropped:
        dropped_path = Path(LEIKA_DATA_DIR).joinpath("dropped_codes.txt")
        with open(dropped_path, "w") as f:
            for code in dropped:
                f.write(f"{code}\n")


if __name__ == "__main__":

    if not LEIKA_CLEANED_DESCS_PATH.exists():
        _clean_json()

    _split()

    sys.exit(0)
