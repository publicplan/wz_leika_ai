import logging
import time
import urllib
from typing import Any, Dict, List, Optional

from elasticsearch import Elasticsearch
from elasticsearch.helpers import ScanError, bulk, scan

ELASTIC_HOST_ENV_VAR = "ELASTIC_HOST"
ELASTIC_PORT_ENV_VAR = "ELASTIC_PORT"
ELASTIC_INDEX_ENV_VAR = "ELASTIC_INDEX"
ELASTIC_QUERY_ENV_VAR = "ELASTIC_QUERY"

logger = logging.getLogger(__name__)


def connect_client(host: str = "localhost", port: int = 9200) -> Elasticsearch:
    """Connect to Elasticsearch instance on host:port."""
    return Elasticsearch([{"host": host, "port": port}])


def retrieve_index(client: Elasticsearch, index_name: str,
                   query: str) -> List[Dict[str, str]]:
    """Retrieve documents in given index.

    Args:
        client: Client connected to Elasticsearch instance.
        index_name: Name of index to retrieve.
        query: Filter documents matching this query. Expected to be in
            Lucene query string syntax.

    Returns:
        Document collection in index (stripped of Elasticsearch metadata).

    Raises:
        LookupError if index was not found or no documents matched the query.
    """
    if not client.indices.exists(index=index_name):
        raise LookupError(f"Index {index_name} does not exist.")
    results = scan(client, q=query, index=index_name)
    if not results:
        raise LookupError(
            f"No documents matching {query} in index {index_name}.")
    return [result["_source"] for result in results]


def fill_index(docs: List[Dict[str, Any]],
               client: Elasticsearch,
               index_name: str,
               doc_type: str,
               id_field: Optional[str] = None) -> None:
    """Fill index with given documents.

    Creates index if it does not exist. Otherwise, append documents to index,
    (possibly overwriting existing ones).


    Args:
        docs: List of documents to upload.
        client: Client connected to Elasticsearch instance.
        index_name: Name of index to fill.
        doc_type: Document type to give documents.
        id_field: If not None, use this field in document collection as ids.
    """

    if id_field is None:
        ids = iter(range(len(docs)))
    else:
        ids = (int(doc[id_field]) for doc in docs)

    client.indices.create(index=index_name, ignore=400)
    request_gen = (_create_request(doc, index_name, idx, doc_type)
                   for idx, doc in zip(ids, docs))
    bulk(client, request_gen)
    client.indices.refresh(index_name)


def _create_request(desc: Dict[str, str], index_name: str, idx: int,
                    doc_type: str):
    return {
        "_index": index_name,
        "_id": idx,
        "_type": doc_type,
        "_source": desc
    }


def elasticsearch_available(host: str, port: int) -> bool:
    """Test if Elasticsearch instance is available on host:port."""
    try:
        health_url = f"http://{host}:{port}/_cluster/health"
        response = urllib.request.urlopen(health_url)  # type: ignore
    except (ConnectionRefusedError, urllib.error.URLError):  # type: ignore
        return False
    else:
        return response.getcode() == 200  # type: ignore


def wait_for_elasticsearch(host: str,
                           port: int,
                           retries: int = 10,
                           timeout: int = 5) -> Elasticsearch:
    """Try to connect to Elasticsearch instance.

    Args:
        host: Host address.
        port: Port number.
        retries: Number of connection attempts before giving up.
        timeout: Time (in seconds) between connection attempts.

    Returns:
        Elasticsearch client.

    Raises:
        ConnectionError if connection could not be established after exhausting
        retries.
    """
    for _ in range(retries):
        logger.info("Trying to connect to Elasticsearch...")
        if elasticsearch_available(host, port):
            logger.info(
                f"Connected to Elasticsearch running on {host}:{port}.")
            return connect_client(host, port)
        time.sleep(timeout)
    raise ConnectionError(
        f"Unable to connect to Elasticsearch on {host}:{port}")


def wait_for_index(client: Elasticsearch,
                   index_name: str,
                   query: str,
                   retries: int = 10,
                   timeout: int = 5) -> List[Dict[str, Any]]:
    """Try to retrieve data from index.

    Args:
        client: Elasticsearch client.
        index_name: Name of index to search.
        query: Filter documents matching this query. Expected to be in
            Lucene query string syntax.
        retries: Number of retrieval attempts before giving up.
        timeout: Time (in seconds) between retrieval attempts.

    Returns:
        Source data in index.

    Raises:
        LookupError if index was not found or no documents matched the query.
    """
    for _ in range(1, retries):
        logger.info(f"Trying to retrieve data from {index_name}...")
        if client.indices.exists(index_name):
            try:
                docs = retrieve_index(client, index_name, query)
            except ScanError:
                pass
            else:
                if docs:
                    # Wait until all docs are uploaded
                    docs = _wait_until_complete(client, index_name, query)
                    logger.info(f"Retrieved {len(docs)} documents.")
                    return docs
            client.indices.refresh(index_name)
        time.sleep(timeout)

    raise LookupError(f"No documents matching {query} in index {index_name}.")


def _wait_until_complete(client: Elasticsearch,
                         index_name: str,
                         query: str,
                         timeout: int = 1) -> List[Dict[str, str]]:
    num_retrieved = 0
    while True:
        docs = retrieve_index(client, index_name, query)
        if num_retrieved and num_retrieved == len(docs):
            return docs
        num_retrieved = len(docs)
        time.sleep(timeout)
