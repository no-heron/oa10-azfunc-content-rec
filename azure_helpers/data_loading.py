import logging, random
from typing import Optional, List
import numpy as np
import pandas as pd

import azure_helpers.cosmos_articles_repository as articles_db
import azure_helpers.cosmos_clicks_repository as clicks_db

# ---------------------------------------------------------------------
# Core Interaction Functions
# ---------------------------------------------------------------------

def get_interactions(clicks_df = None) -> pd.DataFrame:
    """
    Retrieve user-article interaction data from the clicks repository,
    enriched with recency-based weights and timestamps.

    Returns:
        pd.DataFrame: Columns include user_id, article_id, click_time,
                      click_days_ago, recency_weight, etc.
    """
    try:
        click_stats = clicks_db.get_all_clicks() if clicks_df is None else clicks_df.copy()

        if click_stats.empty:
            logging.info("No click data found in get_interactions().")
            return pd.DataFrame(columns=[
                "user_id", "article_id", "click_time", "click_days_ago", "recency_weight"
            ])
        click_stats["click_time"] = pd.to_datetime(click_stats["click_timestamp"], unit="ms")
        max_time = click_stats["click_time"].max()

        click_stats["click_days_ago"] = (max_time - click_stats["click_time"]).dt.days
        click_stats["recency_weight"] = 1 / (1 + click_stats["click_days_ago"])
        click_stats = click_stats.rename(columns={"click_article_id": "article_id"})

        logging.debug("Processed %d user-article interactions.", len(click_stats))
        return click_stats

    except Exception as e:
        logging.exception("Error in get_interactions(): %s", e)
        raise


def get_user_article_affinity_ratings(interactions_df = None) -> pd.DataFrame:
    """
    Compute normalized user-article interaction strengths (ratings) based on
    click frequency and recency.

    Returns:
        pd.DataFrame: Columns [user_id, item_id, rating]
                      where rating ∈ [1, 5].
    """
    try:
        df = get_interactions() if interactions_df is None else interactions_df.copy() 
        if df.empty:
            logging.info("No interactions found for affinity computation.")
            return pd.DataFrame(columns=["user_id", "item_id", "rating"])

        df = (
            df.drop(columns=["click_timestamp"])
            .groupby(["user_id", "article_id"], as_index=False)
            .agg(click_count=("article_id", "count"),
                 recency_weight=("recency_weight", "sum"))
        )

        df["interaction_weight"] = df["click_count"] ** 0.75 + df["recency_weight"] * 3
        # Nonlinear scaling
        strength = np.log1p(df["interaction_weight"]) ** 1.2  # light exponential stretch

        # Rank-normalize to [1, 5]
        ranks = strength.rank(pct=True) # type: ignore
        ratings = 1 + 4 * ranks

        rating_triplets = (
            df.assign(rating=ratings)
            .rename(columns={"article_id": "item_id"})
            [["user_id", "item_id", "rating"]]
        )
        rating_triplets['user_id'] = rating_triplets['user_id'].astype(int)
        rating_triplets['item_id'] = rating_triplets['item_id'].astype(int)

        logging.debug("Generated %d user-article ratings.", len(rating_triplets))
        return rating_triplets

    except Exception as e:
        logging.exception("Error computing user-article affinity ratings: %s", e)
        raise


def get_articles_scores() -> pd.DataFrame:
    """
    Compute freshness and popularity scores for all articles.

    Returns:
        pd.DataFrame: Columns [article_id, freshness_score, popularity_score]
    """
    try:
        articles = articles_db.get_all_articles()
        if articles.empty:
            logging.info("No article data found in get_articles_scores().")
            return pd.DataFrame(columns=["article_id", "freshness_score", "popularity_score"])

        # Compute freshness decay (half-life: 100 days)
        max_ts = articles["created_at_ts"].max()
        decay_rate = 100 * 24 * 3600 * 1000  # ms in 100 days
        articles["freshness_score"] = np.exp(-(max_ts - articles["created_at_ts"]) / decay_rate)

        # Aggregate popularity from interactions
        click_stats = get_interactions()
        if not click_stats.empty:
            popularity = (
                click_stats.groupby("article_id", as_index=False)["recency_weight"].sum()
            )
            popularity["popularity_score"] = (
                np.log1p(popularity["recency_weight"]) /
                np.log1p(popularity["recency_weight"].max())
            )
        else:
            logging.info("No click data found for popularity scoring.")
            popularity = pd.DataFrame(columns=["article_id", "popularity_score"])

        data = (
            articles.merge(
                popularity[["article_id", "popularity_score"]],
                on="article_id",
                how="left"
            )[["article_id", "freshness_score", "popularity_score"]]
        )

        data["popularity_score"] = data["popularity_score"].fillna(0)
        logging.debug("Computed article scores for %d articles.", len(data))
        return data

    except Exception as e:
        logging.exception("Error in get_articles_scores(): %s", e)
        raise


def get_random_users(n_users) -> List[int]:
    all_users = clicks_db.get_users()
    return random.sample(all_users, min(n_users, len(all_users)))

# ---------------------------------------------------------------------
# Pass-through Repository Accessors
# ---------------------------------------------------------------------
def get_clicked_articles_by_user(user_id: int):
    """Wrapper for clicks_repository.get_clicked_articles_by_user()."""
    if not isinstance(user_id, int):
        logging.warning("Invalid user_id in get_clicked_articles_by_user: %s", user_id)
        return []
    return clicks_db.get_clicked_articles_by_user(int(user_id))


def get_last_clicked_by_user(user_id: int) -> Optional[int]:
    """Wrapper for clicks_repository.get_last_clicked_by_user()."""
    if not isinstance(user_id, int):
        logging.warning("Invalid user_id in get_last_clicked_by_user: %s", user_id)
        return None
    return clicks_db.get_last_clicked_by_user(int(user_id))


def get_all_articles() -> pd.DataFrame:
    """Wrapper for articles_repository.get_all_articles()."""
    return articles_db.get_all_articles()