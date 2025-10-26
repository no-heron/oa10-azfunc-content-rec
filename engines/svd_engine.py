from typing import List, Optional

import numpy as np

from azure_helpers.blob_utils import load_model_from_blob_storage
from function_app_logging import get_logger
logger = get_logger("svdpp_engine")

class SVDRecommendationEngine:
    """
    SVD++ collaborative filtering engine for personalized article recommendations.
    """

    def __init__(self, model_path, storage_mode='blob'):
        """
        Initialize the model. If a trained model exists, it is loaded;
        otherwise, a new model is trained and saved.
        """
        logger.info("Initializing SVDRecommendationEngine... Loading model.")
        self.model, self.trainset = self._load_model(model_path, storage_mode)


    def _load_model(self, file_path, storage_mode='blob'):
        """
        Load a trained SVD++ model and its training mappings from a pickle file.
        """
        if storage_mode == 'blob':
            try:
                artifact = load_model_from_blob_storage(blob_name=file_path)
                model = artifact["model"]
                trainset = artifact.get("trainset")

                if trainset is None:
                    logger.warning("Model artifact missing trainset; predictions may be limited.")
                return model, trainset
            
            except Exception as e:
                logger.exception("Error loading SVD++ model from (%s) %s: %s", storage_mode, file_path, e)
                raise
        else:
            raise
    # -------------------------------------------------------------------------
    # Recommendation Logic
    # -------------------------------------------------------------------------

    def recommend_for_user(self, user_id: int, candidates: List[int], N: Optional[int] = None):
        """
        Generate SVD++ predictions for a user across a list of candidate articles.

        Args:
            user_id (int): The user ID for whom to recommend.
            candidates (List[int]): List of candidate article IDs.
            N (Optional[int]): Number of top recommendations to return.

        Returns:
            List[Tuple[int, float]]: (item_id, normalized_score) sorted descending.
        """
        if not candidates:
            logger.info("No candidate items provided for user %s.", user_id)
            return []

        if not self.model or not hasattr(self.model, "predict"):
            logger.error("SVD++ model not initialized.")
            return []
        
       #  print("Types before inferences - User:", type(user_id), "- Article:", {type(candidates[0])})
        known_candidates = []
        for iid in candidates:
            try:
                _ = self.model.trainset.to_inner_iid(iid)  # verify presence
                known_candidates.append(iid)
            except ValueError:
                logger.error("Candidate item %s not in training set; skipping.", iid)

        if not known_candidates:
            logger.info(f'No known candidate items for user {user_id}')
            return []
        else:
            logger.debug(f'Found {len(known_candidates)} candidates.')
        try:
            preds = [
                (iid, self.model.predict(user_id, iid).est)
                for iid in known_candidates
            ]
            scores = np.array([score for _, score in preds])
            if scores.size == 0:
                return []
            # Normalize scores to [0, 1]
            min_s, max_s = scores.min(), scores.max()
            norm_scores = (scores - min_s) / (max_s - min_s) if max_s > min_s else np.zeros_like(scores)

            results = list(zip([iid for iid, _ in preds], norm_scores))
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:N] if N else results

        except Exception as e:
            logger.exception("Error generating SVD++ recommendations for user %s: %s", user_id, e)