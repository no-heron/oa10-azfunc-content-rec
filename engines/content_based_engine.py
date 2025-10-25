import os
import logging
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

from azure_helpers.blob_utils import load_model_from_blob_storage
import azure_helpers.data_loading as db


class ContentBasedRecommendationEngine:
    """
    Content-based recommender using precomputed article embeddings.
    Computes cosine similarity between article vectors for recommendations.
    """

    def __init__(self, embeddings_path, storage_mode='blob'):
        """
        Initialize the recommendation engine by loading article embeddings and metadata.
        """
        logging.info("Initializing ContentBasedRecommendationEngine...")
        try:
            if storage_mode == 'blob':
                embeddings = load_model_from_blob_storage(blob_name=embeddings_path)
            else:
                raise
        except:
            logging.error("Copuld not load embeddings.")
            raise FileNotFoundError("Embeddings file not found or path not set.")
        
        try:
            available_articles = (
                db.get_all_articles()["article_id"]
                .dropna()
                .sort_values()
                .unique()
                .tolist()
            )
            if not available_articles:
                raise ValueError("No available articles found in database.")
        except Exception as e:
            logging.error("No articles found in database query result.")
            raise
        
        try:
            embeddings = embeddings[available_articles]
            self.article_ids = np.array(available_articles)
            self.article_ids_to_index = {aid: idx for idx, aid in enumerate(self.article_ids)}
        except Exception as e:
            logging.exception("Error filtering embeddings: %s", e)
            raise

        try:
            # Normalize embeddings once for cosine similarity via dot product
            self.embeddings = normalize(embeddings, axis=1)
        except Exception as e:
            logging.exception("Error normalizing embeddings: %s", e)
            raise

    # -------------------------------------------------------------------------
    # Recommendation Logic
    # -------------------------------------------------------------------------

    def recommend(self, article_id: int, n_recs: Optional[int] = None) -> List[Tuple[int, float]]:
        """
        Recommend articles similar to the given one, based on cosine similarity.

        Args:
            article_id (int): ID of the reference article.
            n_recs (Optional[int]): Number of recommendations to return (default: all).

        Returns:
            List[Tuple[int, float]]: [(recommended_article_id, similarity_score), ...]
        """
        if article_id not in self.article_ids_to_index:
            logging.warning("Article ID %s not found in embeddings index.", article_id)
            return []

        try:
            article_idx = self.article_ids_to_index[article_id]
            q = self.embeddings[article_idx]
            q = q / np.linalg.norm(q)  # re-normalize query just in case
            sims = self.embeddings @ q  # cosine similarity since pre-normalized

            # Map cosine similarity [-1, 1] → [0, 1]
            sims = (sims + 1) / 2
            sims[article_idx] = -1.0  # exclude the article itself

            # Sort and select top results
            sorted_idx = np.argsort(-sims)
            if n_recs is not None:
                sorted_idx = sorted_idx[:n_recs]

            recs = [(int(self.article_ids[i]), float(sims[i])) for i in sorted_idx]

            logging.debug("Generated %d recommendations for article_id=%d.", len(recs), article_id)
            return recs

        except Exception as e:
            logging.exception("Error computing recommendations for article %s: %s", article_id, e)
            raise