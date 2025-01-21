import json
import logging
import os
import tempfile
import time
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Union, DefaultDict

import click
import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator  # pylint: disable=no-name-in-module

from publicplan.models.classifiers import Classifier
from publicplan.models.config import ModelConfig
from publicplan.paths import WZ_KEYWORDS_PATH, WZ_WEIGHTS_DIR, WZ_GP2019A_PATH

from .data_processing import ihk_keywords
from .description import WZCode, WZDesc, WZDescriptions, build_descriptions, build_gp2019a
from .editing import WZAddition, parse_additions, update_descs
from .keyword_update import (destatis_keywords_date, download_keyword_list,
                             saved_keywords_date)

logger = logging.getLogger(__name__)

LOG_DIR = "api_log"
EDITING_URL_ENV_VAR = "EDITING_URL"


# pylint: disable=no-self-argument,no-self-use
class LogAddition(BaseModel):
    """POST data for additional log route."""

    query_id: str
    source: str
    user_action: str

    no_match: ClassVar[str] = "no match"

    @validator("*")
    def non_empty(cls, v: str, field: str) -> str:
        if not v:
            raise ValueError(f"{field} must be non-empty.")
        return v


class SearchResult(BaseModel):
    """Format for single search result."""
    occupation: WZDesc
    added: Optional[WZAddition]
    ihk_keywords: List[str]
    gp2019a: List[str]
    relevance_confidence: float


class SearchResponse(BaseModel):
    """Search response format."""
    query_id: str
    source: str
    results: List[SearchResult]


class HTTPError(BaseModel):
    """General HTTP Error."""
    detail: str


def _error_responses(
        status_codes: List[int]) -> Dict[Union[int, str], Dict[str, Any]]:
    return {code: {"model": HTTPError} for code in status_codes}


# pylint: disable=too-many-statements,unused-variable
def build_wz_api(clf: Classifier, descs: WZDescriptions,
                 ihk_kws: DefaultDict[WZCode, List[str]],
                 gp2019a: DefaultDict[WZCode, List[str]],
                 accepted_additions: Dict[WZCode, WZAddition],
                 rejected_additions: List[Dict[str, Any]],
                 log_dir: Path) -> FastAPI:
    """Build fastapi app for WZ codes.

    Answers GET requests using predictions of classifier.

    Args:
        clf: Classifier to use for predictions.
        descs: Collection of WZ description.
        ihk_kws: Keywords extracted from IHK data.
        gp2019a: Keywords extracted from GP2019A data.
        accepted_additions: Parsed additions from editing system.
        rejected_additions: Rejected additions from editing system.
        log_dir: Directory to use for logging.

    Returns:
        Constructed fastapi app.
    """

    logging_disabled = False
    try:
        log_dir.mkdir(exist_ok=True, parents=True)
    except OSError as e:
        logger.warning(f"Can not create log directory {log_dir.absolute()}:")
        logger.warning(str(e))
        logger.warning("Logging is disabled.")
        logging_disabled = True

    additions: Dict[WZCode, Optional[WZAddition]] = defaultdict(
        lambda: None, accepted_additions)
    app = FastAPI()

    @app.get("/")
    async def base():
        return "WZ-AI api running."

    @app.get("/suche/", response_model=SearchResponse)
    async def query_search(query: str,
                           num_returned: int = 5,
                           source: str = "debug",
                           user_token: str = "",
                           no_logging: bool = False):
        """Access point for query requests.

        Args:
            query: Search string
            num_returned: Number of candidate codes to return.
            source: Source of the API call (for logging purposes).
            user_token: User identifier (for logging purposes).
            no_logging: Disable query logging.

        Returns:
            Response as application/json, giving the most relevant codes
            together with their description and relevance scores.
        """

        preds = clf.predict(query, num_returned)

        codes = preds.index

        if not source:
            source = "unknown_source"

        if not user_token:
            user_token = "unknown_user"

        timestamp = datetime.now()
        log_path: Optional[Path] = None
        query_id = ""
        if not logging_disabled and not no_logging:
            query_id = uuid.uuid4().hex
            log_path = log_dir.joinpath(source, f"{query_id}.json")
            while log_path.exists():
                query_id = uuid.uuid4().hex
                log_path = log_dir.joinpath(source, f"{query_id}.json")
        results = [
            SearchResult(occupation=descs[code].dict(),
                         ihk_keywords=ihk_kws[descs[code].code],
                         gp2019a=gp2019a[descs[code].code],
                         added=additions[descs[code].code],
                         relevance_confidence=float(preds[code]))
            for code in codes
        ]

        if log_path is not None:
            try:
                log_path.parent.mkdir(exist_ok=True)
                log_results(results,
                            query,
                            timestamp,
                            log_path=log_path,
                            source=source,
                            user_token=user_token)

            # Catch all exceptions to avoid crashing at all costs
            # pylint: disable=broad-except
            except Exception as e:
                logger.warning("Writing log failed:")
                logger.warning(str(e))

        response = {"query_id": query_id, "source": source, "results": results}
        return response

    @app.post("/log_add/",
              response_model=str,
              responses=_error_responses([404, 422]))
    async def log_add(add: LogAddition):
        """Post route for logging of subsequent user actions."""
        user_action = add.user_action
        query_id = add.query_id
        source = add.source

        log_path = log_dir.joinpath(source, f"{query_id}.json")
        if not log_path.exists():
            raise HTTPException(status_code=404,
                                detail=f"Log '{source}/{query_id}' not found")

        content = json.load(open(log_path, "r"))
        results = content["results"]

        if not user_action == LogAddition.no_match:
            displayed_codes = [
                result["occupation"]["code"] for result in results
            ]
            if not user_action in displayed_codes:
                detail = f"Given code ({user_action}) not in "
                detail += f"list of displayed codes ({displayed_codes})."
                raise HTTPException(status_code=422, detail=detail)

        content["user_action"] = user_action
        json.dump(content, open(log_path, "w"), indent=4)

        return f"Successfully updated log '{source}/{query_id}'."

    @app.get("/descs/", response_model=List[WZDesc])
    async def all_descs():
        """List of all available WZ codes with descriptions.

        Returns:
            List of description as application/json response.
        """
        return [descs[code].dict() for code in descs.codes]

    @app.get("/descs/{code}",
             response_model=WZDesc,
             responses=_error_responses([404, 422]))
    async def desc_from_code(code: str):
        """Look up desc from code.

        Returns:
            Description of Code.

        Raises:
            422 Error if code could not be parsed.
            404 Error if code is not in collection.
        """
        try:
            parsed_code = WZCode.parse(code)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        try:
            desc = descs[parsed_code]
        except KeyError:
            raise HTTPException(
                status_code=404,
                detail=f"WZ-Code {code} not found in collection.")

        return desc.dict()

    @app.get("/editing/accepted", response_model=Dict[str, WZAddition])
    async def editing_accepted():
        """Additions from editing sytem"""
        return {
            code: added.dict()
            for code, added in accepted_additions.items()
        }

    @app.get("/editing/rejected")
    async def editing_rejected():
        """Rejected entries from editing sytem"""
        return rejected_additions

    return app


def log_results(results: List[SearchResult], query: str, timestamp: datetime,
                log_path: Path, source: str, user_token: str) -> None:
    """Log api results to json file.

    Args:
        results: Results to log.
        query: API query.
        query: Generated id of query request.
        timestamp: Time of query request.
        log_path: Path of log file.
        source: Source of query.
    """

    log = {
        "query": query,
        "user_token": user_token,
        "source": source,
        "timestamp": timestamp.strftime("%Y%m%d-%H:%M:%S"),
        "results": [result.dict() for result in results]
    }
    json.dump(log, open(log_path, "w"), indent=4)
    logger.info(f"Wrote query log to {log_path.absolute()}")


def _get_additions(
        url: str = "",
        retries: int = 10,
        verify: bool = True) -> Tuple[List[WZAddition], List[Dict[str, Any]]]:
    """Retrieve additions from url."""
    if not url:
        try:
            url = os.environ[EDITING_URL_ENV_VAR]
        except KeyError:
            raise ValueError(
                f"Can not retrieve url for editing system from {EDITING_URL_ENV_VAR}."
            )
    logger.info(f"Retrieving editing additions from {url}.")
    if "https" in url and not verify:
        logger.info(f"Not checking certificate for {url}.")
    for n in range(1, retries + 1):
        try:
            # Publicplan requested to disable verification, at least temporarily
            response = requests.get(url, verify=verify)
            response.raise_for_status()
            break
        except (requests.exceptions.HTTPError,
                requests.exceptions.ConnectionError):
            time.sleep(5)
            logger.warning(
                f"Can not retrieve data from {url} (Try {n}/{retries}).")
    else:
        raise ValueError(f"Data from {url} unavailable.")

    results = response.json()
    if not isinstance(results, list):
        raise ValueError(
            f"Results retrieved from {url} is not a list of additions.")

    accepted, rejected = parse_additions(results)

    return accepted, rejected


def _update_keywords() -> Path:
    keywords_date = destatis_keywords_date()
    old_keywords_date = saved_keywords_date()

    if keywords_date > old_keywords_date:
        logger.info("Found newer keyword list (%s > %s).",
                    keywords_date.strftime("%Y-%m-%d"),
                    old_keywords_date.strftime("%Y-%m-%d"))
        logger.info("Downloading new keywords.")
        keywords_path = download_keyword_list()

        return keywords_path
    logger.info("Keyword list up-to-date.")
    return WZ_KEYWORDS_PATH


@click.command()
@click.option("--host", "-h", type=str, default="0.0.0.0", help="Host address")
@click.option("--port", "-p", type=int, default=8000, help="Port number")
@click.option("--log-dir",
              type=click.Path(),
              default="",
              help="Logging directory")
@click.option("--editing-disable-verification",
              is_flag=True,
              help="Disable certificate verification for editing url")
def cli(host: str, port: int, log_dir: str,
        editing_disable_verification: bool):
    """Start WZ api with uvicorn."""

    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s (%(name)s): %(message)s")

    checkpoint = WZ_WEIGHTS_DIR.joinpath("bert_model")

    try:
        keywords_path = _update_keywords()
    except (ValueError, ConnectionError) as e:
        logger.warning("Error in keyword update:")
        logger.warning(str(e))
        logger.warning("Using old keyword list.")
        keywords_path = WZ_KEYWORDS_PATH
    except Exception as e:  # pylint: disable=broad-except
        logger.error("Unexpted error in keyword update:")
        logger.error(str(e))
        logger.error("Using old keyword list.")
        keywords_path = WZ_KEYWORDS_PATH

    logger.info("Loading descriptions.")
    model_config = ModelConfig.from_checkpoint(checkpoint)

    descs = build_descriptions(keywords_path=keywords_path)
    try:
        additions, rejected = _get_additions(
            verify=not editing_disable_verification)
    except ValueError as e:
        logger.warning("Error in retrieval of editing data.")
        logger.warning(str(e))
        logger.warning("Not loading additions from editing system.")
        additions, rejected = [], []

    gp2019a = build_gp2019a(keywords_path=WZ_GP2019A_PATH)

    new_descs, accepted, rejected2 = update_descs(descs, additions)
    rejected += [r.dict() for r in rejected2]
    ihk_kws = ihk_keywords()
    for code, kws in ihk_keywords().items():
        new_descs[code].keywords += kws
    for code, kws in gp2019a.items():
        new_descs[code].keywords += kws
    clf = model_config.build_model(new_descs)
    if not log_dir:
        log_path = Path(tempfile.gettempdir()).joinpath("wz-api-logs")
    else:
        log_path = Path(log_dir)

    logger.info(f"Using logging dir {log_path.absolute()}")
    logger.info("Starting App.")

    uvicorn.run(build_wz_api(clf,
                             descs,
                             ihk_kws=ihk_kws,
                             gp2019a=gp2019a,
                             accepted_additions=accepted,
                             rejected_additions=rejected,
                             log_dir=log_path),
                host=host,
                port=port)
