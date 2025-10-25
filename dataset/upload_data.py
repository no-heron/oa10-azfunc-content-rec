import os

from azure.cosmos import CosmosClient, PartitionKey
from dotenv import load_dotenv
import pandas as pd


load_dotenv(override=True)
# --- Config ---
COSMOS_CONNECTION_STRING = str(os.getenv("CosmosDbConnectionString"))
DATABASE_NAME = "bookrec"
ARTICLES_CONTAINER = "articles"
CLICKS_CONTAINER = "clicks"


client = CosmosClient.from_connection_string(conn_str=COSMOS_CONNECTION_STRING)
# Create database if it doesn't exist
database = client.create_database_if_not_exists(id=DATABASE_NAME)

# --- Upload Function ---
def upload(container_name, df, id_field):

    db = client.get_database_client(DATABASE_NAME)
    container = db.get_container_client(container_name)

    for _, row in df.iterrows():
        item = row.to_dict()
        item["id"] = str(item[id_field])
        container.upsert_item(item)
    print("Upload complete.")

clicks_container = database.create_container_if_not_exists(
    id=CLICKS_CONTAINER,
    partition_key=PartitionKey(path="/user_id")
)
clicks_df = pd.read_csv('dataset/clicks_sample.csv')
clicks_df['id'] = (
    clicks_df['user_id'].astype(str)
    + '-' +
    clicks_df['session_id'].astype(str)
    + '-' +
    clicks_df['click_timestamp'].astype(str)
)
print("Clicks : uploading", clicks_df.shape[0], "rows and", clicks_df.shape[1], "columns...")
print(clicks_df.head())
upload(CLICKS_CONTAINER, clicks_df, id_field='id')


# --- ARTICLES : run upload ---
articles_container = database.create_container_if_not_exists(
    id=ARTICLES_CONTAINER,
    partition_key=PartitionKey(path="/article_id")
)
articles_df = pd.read_csv('dataset/articles_metadata.csv').drop(columns=['publisher_id']).drop_duplicates(subset='article_id')
articles_df = articles_df.loc[articles_df['article_id'].isin(clicks_df['click_article_id'])]
print("Articles : uploading", articles_df.shape[0], "rows and", articles_df.shape[1], "columns...")
upload(ARTICLES_CONTAINER, articles_df, id_field='article_id')
exit()




