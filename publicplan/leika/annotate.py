import csv
import json
import readline
import sys
from pathlib import Path
from pprint import pprint
from typing import Dict, List, Tuple, Union

import click

from publicplan.models.classifiers import Classifier
from publicplan.models.config import ModelConfig
from publicplan.paths import LEIKA_CLEANED_DESCS_PATH, LEIKA_WEIGHTS_DIR

from .description import LeikaDesc, LeikaDescriptions


def get_suggestions(query: str, num_returned: int, relconf: float,
                    clf: Classifier,
                    descs: LeikaDescriptions) -> List[Tuple[LeikaDesc, float]]:
    """Get suggestions from classifier for given query.

    Args:
        query: Query to consider.
        clf: Classifier to use
        num_returned: Number of returned results.
        relconf: Confidence threshold for inclusion in suggestion.
        descs: Code descriptions to use.

    Returns:
        List of ranked suggestions, consisting of code descriptions
        and confidence score.
    """
    preds = clf.predict(query, num_returned)
    codes = preds.index
    results = [(descs[code], preds[code]) for code in codes
               if preds[code] > relconf]
    return results


def _load(dir_path: Path, file_name: str) -> List[dict]:
    path = dir_path.joinpath(file_name)
    if path.suffix in {".tmp", ".json"}:
        with open(path, "r") as f:
            result: List[dict] = json.load(f)
            return result
    if path.suffix == ".csv":
        with open(path, "r") as f:
            csv_reader = csv.reader(f)
            return [{
                "query": line[0],
                "codes": line[1:]
            } for line in csv_reader]
    raise ValueError(
        "Input format cannot be processed, must be json or csv file.")


def _print_label(label: Tuple[LeikaDesc, float]) -> None:
    info = vars(label[0])
    info["relevance-confidence"] = label[1]
    pprint(info)
    print("")


def check_labels(query: str, labels: List[Tuple[LeikaDesc, float]],
                 label_type: str) -> List[int]:
    """Presents labels for a given query to user and lets him accept or discard them.

    Args:
        query: Query to consider.
        labels: Labels presented to the user.
        label_type: Origin of label (e.q. suggestion or original annotation).

    Returns:
        Codes corresponding to accepted labels.
    """
    accepted_codes = []
    for i, label in enumerate(labels):
        print("Query: '{}'\n".format(query))
        print("{} {}:".format(label_type, i + 1))
        _print_label(label)
        accept = _get_input("Accept label?", ["y", "n"])
        if accept == "y":
            accepted_codes.append(label[0].code)
    return accepted_codes


def _get_input(prompt: str, answers: List[str]) -> str:
    while True:
        answer = input(prompt + " Type one of " + ", ".join(answers) +
                       ".\n>>> ")
        if answer in answers:
            print("")
            return answer
        print("Invalid answer.\n")


def label_query(query: str, old_codes: List[int], num_returned: int,
                relconf: float, clf: Classifier,
                descs: LeikaDescriptions) -> Dict[str, Union[str, List[int]]]:
    """Present old labels and suggestions and process user choices.

    Args:
        query: Query to consider.
        old_codes: Original annotation of query.
        num_returned: Number of results to present.
        relconf: Confidence threshold for inclusion in suggestion.
        descs: Code descriptions to use.

    Returns:
        Query and chosen labels.
    """
    old_labels = [(descs[code], 1.0) for code in old_codes]
    accepted_old_codes = check_labels(query, old_labels, "Old label")
    suggested_labels = [
        suggestion for suggestion in get_suggestions(query, num_returned,
                                                     relconf, clf, descs)
        if suggestion[0].code not in old_codes
    ]
    accepted_suggested_codes = check_labels(query, suggested_labels,
                                            "Suggestion")
    accepted_codes = accepted_old_codes + accepted_suggested_codes
    label_dict: Dict[str, Union[str, List[int]]] = {
        "query": query,
        "codes": accepted_codes
    }
    return label_dict


def _rlinput(prompt: str, prefill: str = "") -> str:
    readline.set_startup_hook(lambda: readline.insert_text(prefill))
    try:
        return input(prompt)
    finally:
        readline.set_startup_hook()


def _get_path(prompt: str, prefill="") -> Path:
    while True:
        path = Path(_rlinput(prompt + ">>> ", prefill=prefill))
        if path.parent.exists():
            return path
        print("Please input a valid path.")


def _initialize(model: str) -> Tuple[Path, LeikaDescriptions, Classifier]:
    checkpoint = LEIKA_WEIGHTS_DIR.joinpath(model)
    temp_path = Path.cwd()
    if temp_path.joinpath("annotation_tool.tmp").exists():
        temp_path.joinpath("annotation_tool.tmp").unlink()
    print("Loading descriptions.")
    descs = LeikaDescriptions.from_path(LEIKA_CLEANED_DESCS_PATH)
    print("Loading classifier.")
    model_config = ModelConfig.from_checkpoint(checkpoint)
    clf = model_config.build_model(descs)
    print("\n" * 2)
    return temp_path, descs, clf


def _save(new_labels: List[dict], dir_path: Path, file_name: str) -> None:
    path = dir_path.joinpath(file_name)
    if path.exists():
        dict_list = _load(dir_path, file_name)
        dict_list.extend(new_labels)
    else:
        dict_list = new_labels
    with open(path, "w+") as f:
        json.dump(dict_list, f, indent=4)


def _final_save(temp_path: Path, input_path: Path):
    temp = _load(temp_path, "annotation_tool.tmp")
    while True:
        output_path = _get_path(
            "Please specify full output path. (Output is a json-file.)\n",
            prefill=str(input_path.parent))
        if output_path.exists():
            save_type = _get_input(
                "This file already exists. Do you want to overwrite it (o), add new "
                "labels to the end of the file (a) or choose a new path (c)?",
                ["o", "a", "c"])
            if not save_type == "c":
                if save_type == "o":
                    output_path.unlink()
                _save(temp, output_path.parent, output_path.name)
                break
        elif output_path.parent.exists():
            _save(temp, output_path.parent, output_path.name)
            break


def _process_query(manual_queries: bool, query_input: List[dict],
                   start_line: int, count: int, num_returned: int,
                   relconf: float, clf: Classifier, descs: LeikaDescriptions,
                   temp_path: Path):
    if manual_queries:
        query = input("Type a query.\n>>> ")
        old_codes: List[int] = []
    else:
        current = query_input[start_line + count]
        query = current["query"]
        old_codes = [int(code) for code in current["codes"]]
    label_dict = label_query(query, old_codes, num_returned, relconf, clf,
                             descs)
    _save([label_dict], temp_path, "annotation_tool.tmp")


@click.command()
@click.option(
    "--num-returned",
    "-n",
    type=int,
    help=
    "The number of codes suggested. Must be an integer value. Default is 10.",
    default=10)
@click.option(
    "--input-file",
    "-i",
    type=str,
    help=
    "The path for the input file. If none given, user may type queries directly.",
    default="")
@click.option(
    "--start_line",
    "-s",
    type=int,
    help=
    "The line in the input file where to start the labeling. Default is 0.",
    default=0)
@click.option(
    "--relconf",
    type=float,
    help=
    "Specifies minimal relevance-confidence value for suggestions. Default is 0.0.",
    default=0.0)
@click.option(
    "--model",
    "-m",
    type=str,
    help=
    "Name of the model to be used for prediction. Default is 'testing_model'.",
    default="testing_model")
def cli(input_file: str, num_returned: int, start_line: int, relconf: float,
        model: str):
    """Annotate search queries using model predictions."""

    manual_queries = input_file == ""
    input_path = Path(input_file)
    if not manual_queries and not input_path.is_file():
        print("The specified input path does not exist.")
        sys.exit()
    elif manual_queries:
        query_input: List[dict] = []
    else:
        query_input = _load(input_path.parent, input_path.name)

    temp_path, descs, clf = _initialize(model)

    count = 0

    while True:
        _process_query(manual_queries, query_input, start_line, count,
                       num_returned, relconf, clf, descs, temp_path)
        count += 1

        if not manual_queries and count + start_line >= len(query_input):
            print(
                "There are no more queries in the input file. From now on input queries manually."
            )
            manual_queries = True

        print("-" * 200)
        action = _get_input("Next query or quit?", ["n", "q"])

        if action == "q":
            print("You have labeled {} queries.".format(count))
            save_progress = _get_input("Do you want to save?",
                                       ["y", "really no"])
            if save_progress == "y":
                _final_save(temp_path, input_path)
            temp_path.joinpath("annotation_tool.tmp").unlink()
            break
