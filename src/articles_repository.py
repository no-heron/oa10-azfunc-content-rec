import os

import numpy as np
import pandas as pd
from azure.cosmos import CosmosClient


def connect_to_db():
    client = CosmosClient.from_connection_string(os.environ["CosmosDBConnection"])
    database = client.get_database_client("bookrec")
    return database.get_container_client("articles")

def get_all_articles() -> pd.DataFrame:
    c = connect_to_db()
    res = []
    query = "SELECT DISTINCT c.article_id, c.created_at_ts FROM c ORDER BY c.article_id ASC"
    for doc in c.query_items(query=query, enable_cross_partition_query=True):
        res.append({"article_id": doc["article_id"], "created_at_ts": doc["created_at_ts"]})
    return pd.DataFrame(res)

def get_n_newest(n):
    c = connect_to_db()

    res = []
    query = f"SELECT DISTINCT c.article_id, c.created_at_ts FROM c order by created_at_ts desc limit {n}"
    for doc in c.query_items(query=query, enable_cross_partition_query=True):
        res.append({"article_id": doc["article_id"], "created_at_ts": doc["created_at_ts"]})
    new_articles_df = pd.DataFrame(res)
    
    max_ts = new_articles_df['created_at_ts'].max()
    decay_rate = 100 * 24 * 3600 * 1000  # 100 days half-life, in ms
    new_articles_df["freshness_score"] = np.exp(-(max_ts - new_articles_df["created_at_ts"]) / decay_rate)

    return new_articles_df.to_dict(orient="records")