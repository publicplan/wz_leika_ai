import json
import logging
import os
import sys
from typing import Any, Dict, List

from publicplan import elastic as es
from publicplan.paths import LEIKA_DESCS_PATH

logger = logging.getLogger(__name__)


def fill_elastic(docs: List[Dict[str, Any]], index_name: str,
                 doc_type: str) -> int:
    """Connect to elastic instance and fill index with documents."""

    try:
        host = os.environ[es.ELASTIC_HOST_ENV_VAR]
        port = int(os.environ[es.ELASTIC_PORT_ENV_VAR])
    except KeyError as e:
        logger.error(f"Environment variable {str(e)} not found.")
        return 1

    retries = 10
    timeout = 5

    try:
        client = es.wait_for_elasticsearch(host, port, retries, timeout)
    except ConnectionError as e:
        logger.error(str(e))
        return 1

    es.fill_index(docs=docs,
                  client=client,
                  index_name=index_name,
                  doc_type=doc_type)
    logger.info("Added elastic content. Exiting.")
    return 0


def _main():

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s (%(name)s): %(message)s")

    descs = json.load(open(LEIKA_DESCS_PATH, 'r'))
    exit_code = fill_elastic(descs,
                             index_name="default_leika_collection",
                             doc_type="Leika Code description")

    sys.exit(exit_code)


if __name__ == "__main__":
    _main()
