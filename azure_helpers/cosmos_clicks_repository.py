import os
import logging
from typing import List, Optional

import pandas as pd
from azure.cosmos import CosmosClient, exceptions

# ---- Configuration ----
COSMOS_CONNECTION_STRING = os.getenv("CosmosDbConnectionString")
DATABASE_NAME = "bookrec"
CONTAINER_NAME = "clicks"

# ---- Cached client connection ----
_client = None
_container = None

from function_app_logging import get_logger
logger = get_logger("clicks_repo")

def get_container():
    """
    Lazily initialize and cache the Cosmos DB container client.
    """
    global _client, _container
    if _container is not None:
        return _container

    if not COSMOS_CONNECTION_STRING:
        logger.critical("CosmosDbConnectionString environment variable not set.")
        raise RuntimeError("Missing Cosmos DB connection string.")

    try:
        _client = CosmosClient.from_connection_string(COSMOS_CONNECTION_STRING)
        database = _client.get_database_client(DATABASE_NAME)
        _container = database.get_container_client(CONTAINER_NAME)
        logger.info("Connected to Cosmos DB container '%s'.", CONTAINER_NAME)
        return _container
    
    except exceptions.CosmosResourceNotFoundError:
        logger.error("Database or container not found: %s / %s", DATABASE_NAME, CONTAINER_NAME)
        raise
    except Exception as e:
        logger.exception("Unexpected error initializing Cosmos DB connection: %s", e)
        raise


# ---- Query Functions ----
def get_all_clicks() -> pd.DataFrame:
    """
    Retrieve all click records as a pandas DataFrame.
    """
    container = get_container()
    query = "SELECT c.user_id, c.session_id, c.click_article_id, c.click_timestamp FROM c"

    df = pd.DataFrame(columns=['user_id', 'session_id', 'click_article_id', 'click_timestamp'])
    # Assign int64 type to each column
    df = df.astype({
        'user_id': 'int64',
        'session_id': 'int64',
        'click_article_id': 'int64',
        'click_timestamp': 'int64'
    })

    # print("Types received from DB query.\n", df.dtypes)
    try:
        results = list(container.query_items(query, enable_cross_partition_query=True))
        if not results:
            logger.info("No click records found.")
            return df
        return pd.concat([df, pd.DataFrame(results)])

    except exceptions.CosmosHttpResponseError as e:
        logger.error("Cosmos DB query error in get_all_clicks: %s", e)
        raise
    except Exception as e:
        logger.exception("Unexpected error retrieving clicks: %s", e)
        raise


def get_clicked_articles_by_user(user_id: int) -> List[int]:
    """
    Retrieve all clicked article IDs for a given user.
    """
    if not isinstance(user_id, int):
        logger.warning("Invalid user_id provided to get_clicked_articles_by_user: %s", user_id)
        return []
    container = get_container()
    query = "SELECT c.click_article_id FROM c WHERE c.user_id = @user_id"
    params = [{"name": "@user_id", "value": user_id}]

    try:
        items = container.query_items(query=query, parameters=params, enable_cross_partition_query=True)
        articles = [int(doc["click_article_id"]) for doc in items]
        logger.debug("User %s clicked %d articles.", user_id, len(articles))
        return list[int](articles)
    except exceptions.CosmosHttpResponseError as e:
        logger.error("Cosmos DB query error in get_clicked_articles_by_user: %s", e)
        raise
    except Exception as e:
        logger.exception("Unexpected error retrieving user clicks: %s", e)
        raise


def get_last_clicked_by_user(user_id: int) -> Optional[int]:
    """
    Retrieve the most recently clicked article ID for a given user.
    Returns None if no record exists.
    """
    if not isinstance(user_id, int):
        logger.warning("Invalid user_id provided to get_last_clicked_by_user: %s", user_id)
        return None

    container = get_container()
    query = "SELECT TOP 1 c.click_article_id FROM c WHERE c.user_id = @user_id ORDER BY c.click_timestamp DESC"
    params = [{"name": "@user_id", "value": user_id}]

    try:
        results = list(container.query_items(query=query, parameters=params, enable_cross_partition_query=True))
        last_click = int(results[0]["click_article_id"]) if results else None
        logger.debug("User %s last clicked article: %s", user_id, last_click)
        return last_click
    except exceptions.CosmosHttpResponseError as e:
        logger.error("Cosmos DB query error in get_last_clicked_by_user: %s", e)
        raise
    except Exception as e:
        logger.exception("Unexpected error retrieving last click: %s", e)
        raise


def get_users() -> List[int]:
    container = get_container()
    query = "SELECT c.user_id FROM c"
    try :
        results = list(container.query_items(query=query, enable_cross_partition_query=True))
        users = list({int(doc["user_id"]) for doc in results})
        logger.debug(f"{len(users)} found.")
        return users
    except exceptions.CosmosHttpResponseError as e:
        logger.error("Cosmos DB query error in get_last_clicked_by_user: %s", e)
        raise
    except Exception as e:
        logger.exception("Unexpected error retrieving last click: %s", e)
        raise