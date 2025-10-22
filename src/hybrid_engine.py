import logging

import numpy as np
import pandas as pd

from src.content_based_engine import ContentBasedRecommendationEngine as ContentBased
from src.svd_engine import SVDRecommendationEngine as SVDEngine

import src.data_loading as db

class HybridRecommendationEngine():
    def __init__(self, n_recs):

        self.data = db.get_articles_scores()
        self.content_based_engine = ContentBased()
        self.cf_engine = SVDEngine()
        self.n_recs = n_recs

        self.scores = ['freshness_score', 'popularity_score', 'cb_score', 'cf_score']

    def __recommend_popular(self, n_recs:int):
        recs = (
            self.data.sort_values(by='popularity_score', ascending=False)
            .head(n_recs)
        )
        return [(int(article_id), float(row.popularity_score)) for article_id, row in recs.iterrows()] # type: ignore

    def __recommend_new(self, n_recs: int):
        recs = (
            self.data
            .sort_values(by='freshness_score', ascending=False)
            .head(n_recs)
        )
        return [(int(article_id), float(row.freshness_score)) for article_id, row in recs.iterrows()] # type: ignore
 
    def __recommend_content_based(self, article_id):
        ct_based_recs = self.content_based_engine.recommend(article_id)
        if not ct_based_recs:
            return None
        
        cb = (
            pd.DataFrame(ct_based_recs, columns=['article_id', 'cb_score'])
            .sort_values(by='article_id', ascending=True)
        )
        return cb

    def __recommend_collaborative_filtering(self, user_id, n_recs=None):
        data = self.data
        seen = db.get_clicked_articles_by_user(user_id)
        # Exclude articles the user has already seen (index-based)
        candidates = data.loc[~data['article_id'].isin(seen), 'article_id'].to_list()
        recs = self.cf_engine.recommend_for_user(user_id=user_id, candidates=candidates, N=n_recs)
        return recs

    def __get_weights(self, user_id):
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

    def recommend(self, user_id=None, article_id=None):
        logging.debug("Passed arguments: ", user_id, article_id)
        recs = self.data.copy().sort_values(by='article_id', ascending=True)
        
        logging.debug("Popularity & Freshness based recs df shape:", recs.shape)

        if article_id is not None:
            logging.debug(f"\nContent based recommendations based on article {article_id}:")
        elif user_id is not None:
            logging.debug(f"\nContent based recommendations based on last clicked content by user {user_id}:")
            try:
                article_id = db.get_last_clicked_by_user(int(user_id))
            except:
                logging.error(f"Could not retrieve data for user {user_id}")
            
        if article_id:
            cb = self.__recommend_content_based(article_id)
            if cb is not None:
                recs = recs.merge(cb, how='left', on='article_id')
        else:
            logging.debug("No article_id passed or found for specified user.")

        if user_id and db.get_last_clicked_by_user(user_id):
            cf = pd.DataFrame(
                self.__recommend_collaborative_filtering(user_id, None),
                columns=['article_id', 'cf_score']
            ).sort_values(by='article_id', ascending=True)
            recs = recs.merge(cf, how='left', on='article_id')
        else:
            logging.debug("No user_id was passed.")

        
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