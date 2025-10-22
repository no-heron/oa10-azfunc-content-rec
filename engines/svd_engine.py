import os

import numpy as np

from surprise import Dataset, Reader, SVDpp

import data_loading as db

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_file = os.path.join(BASE_DIR, 'models', 'svd', 'svdpp_model.pkl')

class SVDRecommendationEngine:

    def __init__(self, model_file:str=model_file):

        self.model = self.build_train_model(model_file)

        """
        with open(model_file, "rb") as f:
            artifact = pickle.load(f)
            self.model = artifact["model"]
        """
        # user_inner_to_raw = artifact["user_inner_to_raw"]
        # item_inner_to_raw = artifact["item_inner_to_raw"]

    def build_train_model(self, save_model_path=model_file):
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

        """
        with open(save_model_path, "wb") as f:
            print("Saving model to:", f)
            pickle.dump(artifact, f)
        """
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