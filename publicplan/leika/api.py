import json
import logging
import os
import sys
from typing import Any, Dict, List, Type

import click
import uvicorn
from fastapi import FastAPI, HTTPException

from .. import elastic as es
from ..models.classifiers import Classifier
from ..models.config import ModelConfig
from ..paths import LEIKA_CLEANED_DESCS_PATH, LEIKA_WEIGHTS_DIR, SPELLDICT_PATH
from ..spellcheck import SpellChecker
from .description import parse_descs

logger = logging.getLogger(__name__)


def load_desc(retries: int = 10,
              timeout: int = 20,
              fallback: bool = False,
              retry_index: bool = False) -> List[Dict[str, str]]:
    """Load collection of LeiKa descriptions from index of Elasticsearch instance.

    The host, port and index name are read from the environment variables:
    If the query environment is set, then results are restricted to match this query.

    Args:
        retries: Number of attempts to connect to instance.
        timeout: Waiting time for each attempt.
        fallback: If true, use default LeiKa collection if they could not be retrieved
            from instance.
        retry_index: If true, also try multiple times to retrieve non-empty
            collection from index.

    Returns:
        Retrieved collection.

    Raises:
        KeyError if environment variables are not set and fallback is false.
        ConnectionError if unable to connect to instance and fallback is false.
        LookupError if index is not found or is empty and fallback is false.
    """

    try:
        host = os.environ[es.ELASTIC_HOST_ENV_VAR]
        port = int(os.environ[es.ELASTIC_PORT_ENV_VAR])
        index_name = os.environ[es.ELASTIC_INDEX_ENV_VAR]
    except KeyError as e:
        msg = f"Environment variable {str(e)} not found."
        return _handle_exception(msg, KeyError, fallback)
    try:
        query = os.environ[es.ELASTIC_QUERY_ENV_VAR]
    except KeyError as e:
        logger.info(f"Environment variable {str(e)} not found.")
        logger.info("Using wildcard query.")
        query = "*"
    try:
        client = es.wait_for_elasticsearch(host, port, retries, timeout)
    except ConnectionError as e:
        return _handle_exception(str(e), KeyError, fallback)

    try:
        if retry_index:
            descs = es.wait_for_index(client, index_name, query, retries,
                                      timeout)
        else:
            descs = es.retrieve_index(client, index_name, query)
    except LookupError as e:
        return _handle_exception(str(e), LookupError, fallback)

    return descs


def _handle_exception(msg, exception: Type[Exception],
                      fallback: bool) -> List[Dict[str, str]]:

    if not fallback:
        raise exception(msg)
    logger.info(msg)
    logger.info("Using default collection.")
    return json.load(open(LEIKA_CLEANED_DESCS_PATH, 'r'))  #type: ignore


# pylint: disable=unused-variable
def build_leika_api(clf: Classifier, accepted_descs: List[Dict[str, Any]],
                    rejected_descs: List[Dict[str, Any]],
                    spellcheck: SpellChecker) -> FastAPI:
    """Build fastapi app for LeiKa codes.

    Answers GET requests using predictions of classifier.

    Args:
        clf: Classifier to use for predictions.
        accepted_descs: Accepted Leika code descriptions.
        rejected_descs: Malformed entries rejected during parsing.

    Returns:
        Constructed fastapi app.
    """
    descs_by_code = {int(desc["schluessel"]): desc for desc in accepted_descs}

    app = FastAPI()

    @app.get("/")
    async def base():
        return "LeiKa-AI API up and running."

    @app.get("/suche/")
    async def query_search(query: str, num_returned: int = 5):
        """Access point for query requests.

        Answers to GET requests of the form
        /suche/?query=beispielanfrage&num_returned=10

        Args:
            query: Search string
            num_returned: Number of candidate codes to return.

        Returns:
            Response as application/json, giving the most relevant codes
            together with their description and relevance scores.
        """

        preds = clf.predict(query, num_returned)
        codes = preds.index

        results = [{
            "service": descs_by_code[code],
            "relevance-confidence": float(preds[code])
        } for code in codes]

        return results

    @app.get("/codes")
    async def accepted_codes():
        """Return list of Leika code descriptions API is working with."""
        return accepted_descs

    @app.get("/rejected")
    async def rejected_codes():
        """Return list of descriptions which were rejected during parsing."""
        return rejected_descs

    @app.get("/descs/{code}")
    async def desc_from_code(code: int):
        """Look up desc from code.

        Returns:
            Description of Code.

        Raises:
            422 Error if code could not be parsed.
            404 Error if code is not in collection.
        """
        if not code in descs_by_code:
            raise HTTPException(
                status_code=404,
                detail=f"LeiKa-code {code} not found in collection.")
        return descs_by_code[code]

    @app.get("/spellcheck/")
    async def check_query(query: str):
        return spellcheck.correct(query)

    return app


@click.command()
@click.option("--host", "-h", type=str, default="0.0.0.0", help="Host address")
@click.option("--port", "-p", type=int, default=8000, help="Port number")
@click.option("--checkpoint",
              "-c",
              type=str,
              default="shipped_model",
              help="Model directory to use (relative to weights directory)")
@click.option("--fallback",
              "-f",
              is_flag=True,
              default=False,
              help="Use default LeiKa collection as fallback")
@click.option("--retry-index",
              is_flag=True,
              default=True,
              help="Retry retrieving from elasticsearch index")
@click.option("--test-api",
              is_flag=True,
              help="Use smaller model and data for testing purposes")
def cli(host: str, port: int, checkpoint: str, fallback: bool,
        retry_index: bool, test_api: bool):
    """Start Leika api with uvicorn."""

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s (%(name)s): %(message)s")

    weights_dir = LEIKA_WEIGHTS_DIR.joinpath(checkpoint)

    logger.info("Loading descriptions.")
    try:
        raw_descs = load_desc(fallback=fallback, retry_index=retry_index)
    except (ConnectionError, KeyError, LookupError) as e:
        logger.error(str(e))
        logger.error("Aborting.")
        sys.exit(1)

    try:
        parsed_descs, accepted_descs, rejected_descs = parse_descs(raw_descs)
    except ValueError as e:
        logger.error(str(e))
        logger.error("Aborting.")
        sys.exit(1)

    model_config = ModelConfig.from_checkpoint(weights_dir)
    if test_api:
        model_config.params["pretrained_embedding"] = "fasttext_german_pruned"
    clf = model_config.build_model(parsed_descs)

    logger.info("Loading spellchecker.")
    spellchecker = SpellChecker(SPELLDICT_PATH, edit_distance=2)

    logger.info("Starting App.")
    app = build_leika_api(clf, accepted_descs, rejected_descs, spellchecker)
    uvicorn.run(app, host=host, port=port)
