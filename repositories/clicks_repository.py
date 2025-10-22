import os

import numpy as np
import pandas as pd
from azure.cosmos import CosmosClient

client = CosmosClient.from_connection_string(os.environ["CosmosDBConnection"])
database = client.get_database_client("bookrec")
global c
c = database.get_container_client("clicks")


def get_all_clicks() -> pd.DataFrame:
    res = []
    query = f"SELECT DISTINCT c.user_id, c.session_id, c.click_article_id, c.click_timestamp FROM c"
    for doc in c.query_items(query=query, enable_cross_partition_query=True):
        res.append({"user_id": doc["user_id"],
                    "session_id": doc["session_id"],
                    "click_article_id": doc["click_article_id"],
                    "click_timestamp": doc["click_timestamp"]})
    df = pd.DataFrame(res)
    return df


def get_clicked_articles_by_user(user_id) -> list:
    if not user_id:
        return []
    query = f"SELECT c.click_article_id FROM c WHERE c.user_id = {user_id}"
    return [doc["click_article_id"] for doc in c.query_items(
        query=query, enable_cross_partition_query=True
    )]


def get_last_clicked_by_user(user_id: int):
    if not user_id:
        return None
    query = """
        SELECT TOP 1 c.click_article_id
        FROM c
        WHERE c.user_id = @user_id
        ORDER BY c.click_timestamp DESC
    """
    params = [{"name": "@user_id", "value": int(user_id)}]
    results = list(c.query_items(
        query=query,
        parameters=params,
        enable_cross_partition_query=True
    ))
    return results[0]["click_article_id"] if results else None