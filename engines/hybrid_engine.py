import logging
import os

import numpy as np
import pandas as pd

import azure_helpers.data_loading as db
from engines.content_based_engine import ContentBasedRecommendationEngine as ContentBased
from engines.svd_engine import SVDRecommendationEngine as SVDEngine

from function_app_logging import get_logger
logger = get_logger("hybrid_engine")

class HybridRecommendationEngine():
    def __init__(self, n_recs):
        logger.debug("Loading popularity and freshness...")
        self.data = db.get_articles_scores()

        logger.debug("Loading content-based engine...")
        self.content_based_engine = ContentBased(embeddings_path=os.getenv("ArticlesEmbeddingsFile"), storage_mode='blob')

        logger.debug("Loading collaborative filtering SVD++ engine")
        self.cf_engine = SVDEngine(model_path=os.getenv("SVDppModelFile"), storage_mode='blob')
        self.n_recs = n_recs

        self.scores = ['freshness_score', 'popularity_score', 'cb_score', 'cf_score']

    def __recommend_popular(self, n_recs:int):
        logger.debug(f'Issuing recommendations based on popularity...')
        recs = (
            self.data.sort_values(by='popularity_score', ascending=False)
            .head(n_recs)
        )
        return [(str(article_id), float(row.popularity_score)) for article_id, row in recs.iterrows()] # type: ignore

    def __recommend_new(self, n_recs: int):
        logger.debug(f'Issuing recommendations based freshness score...')
        recs = (
            self.data
            .sort_values(by='freshness_score', ascending=False)
            .head(n_recs)
        )
        return [(str(article_id), float(row.freshness_score)) for article_id, row in recs.iterrows()] # type: ignore
 
    def __recommend_content_based(self, article_id):
        logger.debug(f'Issuing recommendations based on article {article_id}')
        ct_based_recs = self.content_based_engine.recommend(article_id)
        if not ct_based_recs:
            return None
        
        cb = (
            pd.DataFrame(ct_based_recs, columns=['article_id', 'cb_score'])
            .sort_values(by='article_id', ascending=True)
        )
        return cb

    def __recommend_collaborative_filtering(self, user_id, n_recs=None):
        logger.debug(f'Issuing recommendations based on collaborative-filtering scores for user {user_id}.')
        data = self.data
        seen = db.get_clicked_articles_by_user(user_id)
        # Exclude articles the user has already seen (index-based)
        candidates = data.loc[~data['article_id'].isin(seen), 'article_id'].to_list()
        recs = self.cf_engine.recommend_for_user(user_id=user_id, candidates=candidates, N=n_recs)
        return recs

    def __get_weights(self, user_id):
        logger.debug('Calculating weights based on user profile...')
        keys = self.scores
        user_history = db.get_clicked_articles_by_user(user_id)
        history_size = len(user_history)

        if history_size > 0:
            cf_weight = 1 / (1 + np.exp(-0.3*(history_size - 8)))  # grows after ~8 clicks
        else :
            cf_weight = 0.0

        cb_weight = min(cf_weight + 0.2, 0.5)
        fresh_pop = max(1 - cf_weight, 0.2)

        w = np.array([fresh_pop/2, fresh_pop/2, cb_weight, cf_weight])
        w = w / w.sum()
        return dict(zip(keys, w))

    def recommend(self, user_id: int | None = None, article_id: int | None = None):
        logger.debug(f"Passed arguments: user_id={user_id}, article_id={article_id}")
        recs = self.data.copy().sort_values(by='article_id', ascending=True)
        if article_id is not None:
            logger.debug(f"\nContent based recommendations based on article {article_id}:")
        elif user_id is not None:
            logger.debug(f"\nContent based recommendations based on last clicked content by user {user_id}:")
            try:
                article_id = db.get_last_clicked_by_user(int(user_id))
            except:
                logger.error(f"Could not retrieve data for user {str(user_id)}")
            
        if article_id:
            cb = self.__recommend_content_based(article_id)
            if cb is not None:
                recs = recs.merge(cb, how='left', on='article_id')
        else:
            logger.debug("No article_id passed or found for specified user.")

        if user_id and db.get_last_clicked_by_user(user_id):
            cf = pd.DataFrame(
                self.__recommend_collaborative_filtering(user_id, None),
                columns=['article_id', 'cf_score']
            ).sort_values(by='article_id', ascending=True)
            recs = recs.merge(cf, how='left', on='article_id')
        else:
            logger.debug("No user_id was passed.")

        recs.fillna(0.0)
        weights = self.__get_weights(user_id)
        weights = {k: v for k, v in weights.items() if k in recs.columns}
        
        w = np.array(list(weights.values()))

        if w.sum() > 0:
            w = w / w.sum()
            weights = dict(zip(weights.keys(), w))

        recs['overall_score'] = (
            recs[list(weights.keys())]
            .mul(pd.Series(weights))
            .sum(axis=1)
        )

        return (
            recs.sort_values(by="overall_score", ascending=False)
                .head(self.n_recs)
                .to_dict(orient="records")
        )