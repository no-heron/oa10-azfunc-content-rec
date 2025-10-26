import os
import logging
from typing import List

import numpy as np
import pandas as pd
from azure.cosmos import CosmosClient, exceptions

# ---- Configuration ----
COSMOS_CONNECTION_STRING = os.getenv("CosmosDbConnectionString")
DATABASE_NAME = "bookrec"
CONTAINER_NAME = "articles"

_client = None
_container = None


def get_container():
    """
    Lazily initialize and cache the Cosmos DB container client.
    """
    global _client, _container
    if _container is not None:
        return _container

    if not COSMOS_CONNECTION_STRING:
        logging.critical("CosmosDbConnectionString environment variable not set.")
        raise RuntimeError("Missing Cosmos DB connection string.")

    try:
        _client = CosmosClient.from_connection_string(COSMOS_CONNECTION_STRING)
        database = _client.get_database_client(DATABASE_NAME)
        _container = database.get_container_client(CONTAINER_NAME)
        logging.info("Connected to Cosmos DB container '%s'.", CONTAINER_NAME)
        return _container
    except exceptions.CosmosResourceNotFoundError:
        logging.error("Database or container not found: %s / %s", DATABASE_NAME, CONTAINER_NAME)
        raise
    except Exception as e:
        logging.exception("Unexpected error initializing Cosmos DB connection: %s", e)
        raise


def get_all_articles() -> pd.DataFrame:
    """
    Retrieve all article metadata from Cosmos DB as a pandas DataFrame.
    """
    container = get_container()
    query = "SELECT c.article_id, c.created_at_ts FROM c ORDER BY c.article_id ASC"

    try:
        items = list(container.query_items(query=query, enable_cross_partition_query=True))
        if not items:
            logging.info("No articles found in container '%s'.", CONTAINER_NAME)
            return pd.DataFrame(columns=["article_id", "created_at_ts"])
        return pd.DataFrame(items)
    except exceptions.CosmosHttpResponseError as e:
        logging.error("Cosmos DB query error in get_all_articles: %s", e)
        raise
    except Exception as e:
        logging.exception("Unexpected error in get_all_articles: %s", e)
        raise


def get_n_newest(n: int) -> List[dict]:
    """
    Retrieve the N newest articles from Cosmos DB, computing a freshness score
    (100-day half-life exponential decay).

    Args:
        n (int): Number of most recent articles to retrieve.

    Returns:
        List[dict]: List of dicts with article_id, created_at_ts, freshness_score.
    """
    if not isinstance(n, int) or n <= 0:
        logging.warning("Invalid argument 'n' to get_n_newest: %s", n)
        return []

    container = get_container()
    query = """
        SELECT c.article_id, c.created_at_ts
        FROM c
        ORDER BY c.created_at_ts DESC
        OFFSET 0 LIMIT @n
    """
    params = [{"name": "@n", "value": n}]

    try:
        items = list(container.query_items(
            query=query,
            parameters=params,
            enable_cross_partition_query=True
        ))

        if not items:
            logging.info("No recent articles found.")
            return []

        df = pd.DataFrame(items)

        # Compute freshness score (half-life: 100 days)
        max_ts = df["created_at_ts"].max()
        decay_rate = 100 * 24 * 3600 * 1000  # ms in 100 days
        df["freshness_score"] = np.exp(-(max_ts - df["created_at_ts"]) / decay_rate)

        logging.debug("Computed freshness scores for %d newest articles.", len(df))
        return df.to_dict(orient="records")

    except exceptions.CosmosHttpResponseError as e:
        logging.error("Cosmos DB query error in get_n_newest: %s", e)
        raise
    except Exception as e:
        logging.exception("Unexpected error in get_n_newest: %s", e)
        raise