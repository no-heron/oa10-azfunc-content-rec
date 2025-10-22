import numpy as np
import pandas as pd

import repositories.articles_repository as articles_db
import repositories.clicks_repository as clicks_db


def get_interactions():
    click_stats = clicks_db.get_all_clicks()
    click_stats['click_time'] = pd.to_datetime(click_stats['click_timestamp'], unit='ms')

    max_time = click_stats['click_time'].max()
    click_stats["click_days_ago"] = (max_time - click_stats['click_time']).dt.days
    click_stats["recency_weight"] = 1 / (1 + click_stats["click_days_ago"])

    click_stats = click_stats.rename(columns={'click_article_id': 'article_id'})
    return click_stats


def get_user_article_affinity_ratings():
    df = get_interactions()
    df = (
        df.drop(columns=['click_timestamp'])
        .groupby(['user_id', 'article_id'])
        .agg(click_count=('article_id', 'count'),
             recency_weight=('recency_weight', 'sum'))
        .reset_index()
    )
    df["interaction_weight"] = (
        df["click_count"] + df["recency_weight"]
    )
    s = df['interaction_weight']
    strength = np.log1p(s)  # smooth heavy tails

    # min-max to [1,5]
    mn, mx = strength.min(), strength.max()
    ratings = 1 + 4 * (strength - mn) / (mx - mn + 1e-9)
    rating_triplets = (
        df.assign(rating=ratings)
        .rename(columns={"article_id":"item_id"})
        [["user_id","item_id","rating"]]
    )
    return rating_triplets


def get_articles_scores():
    articles = articles_db.get_all_articles()
    max_ts = articles['created_at_ts'].max()
    decay_rate = 100 * 24 * 3600 * 1000  # 100 days half-life, in ms
    articles["freshness_score"] = np.exp(-(max_ts - articles["created_at_ts"]) / decay_rate)

    click_stats = get_interactions()
    click_stats = click_stats.groupby("article_id", as_index=False)["recency_weight"].sum()
    click_stats['popularity_score'] = np.log1p(click_stats["recency_weight"]) / np.log1p(click_stats["recency_weight"].max())

    data = (
        articles.merge(
        click_stats, how='left',
        left_on='article_id',
        right_on='article_id'
        )[['article_id', 'freshness_score', 'popularity_score']]
    )
    return data


def get_clicked_articles_by_user(user_id: int):
    return clicks_db.get_clicked_articles_by_user(user_id)


def get_last_clicked_by_user(user_id: int):
    return clicks_db.get_last_clicked_by_user(int(user_id))

def get_all_articles():
    return articles_db.get_all_articles()