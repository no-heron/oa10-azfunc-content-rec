import os
import pickle

import numpy as np
from surprise import Dataset, Reader, SVDpp

from . import data_loading as db

# from dotenv import load_dotenv
# load_dotenv(override=True)

class SVDRecommendationEngine:
    
    def __init__(self): #):
        self.model_file = str(os.getenv("SVDppModelFile"))
        self.model = self._build_train_model(self.model_file)

    def _build_train_model(self, save_model_path):
        train_triplets = db.get_user_article_affinity_ratings()
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(train_triplets, reader)
        trainset = data.build_full_trainset()

        model = SVDpp(
            n_factors=100,     # dimensionality of latent vectors
            n_epochs=20,       # training passes over data
            lr_all=0.004,      # learning rate for all biases and factors
            reg_all=0.04,      # regularization (slightly higher for implicit data)
            random_state=42
        )
        model.fit(trainset)

        # Save model and user/item mappings
        artifact = {
            "model": model,
            "user_raw_to_inner": trainset._raw2inner_id_users,
            "item_raw_to_inner": trainset._raw2inner_id_items,
            "user_inner_to_raw": trainset._inner2raw_id_users,
            "item_inner_to_raw": trainset._inner2raw_id_items,
        }

        with open(save_model_path, "wb") as f:
            print("Saving model to:", f)
            pickle.dump(artifact, f)

        return model
    

    def _load_model(self):
        if not os.path.exists(self.model_file):
            raise FileNotFoundError(f"Model file not found: {self.model_file}")

        with open(self.model_file, "rb") as f:
            model = pickle.load(f)
        return model

    def recommend_for_user(self, user_id, candidates, N=None):
            
        known_candidates = []
        for iid in candidates:
            try:
                _ = self.model.trainset.to_inner_iid(iid)  # check existence
                known_candidates.append(iid)
            except ValueError:
                continue  # item not in model; skip it

        if not known_candidates:
            return []
        # Compute predicted ratings for candidate items
        preds = [(iid, self.model.predict(user_id, iid).est) for iid in known_candidates]

        # Extract scores and normalize them to [0, 1]
        scores = np.array([p[1] for p in preds])
        if len(scores) == 0:
            return []

        min_s, max_s = scores.min(), scores.max()
        if max_s > min_s:
            norm_scores = (scores - min_s) / (max_s - min_s)
        else:
            norm_scores = np.zeros_like(scores)
        # Combine normalized scores with item IDs
        scored = list(zip([iid for iid, _ in preds], norm_scores))
        # Sort by normalized score descending and take top N
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:N] if N is not None else scored