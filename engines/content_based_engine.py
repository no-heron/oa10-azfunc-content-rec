import numpy as np
import pandas as pd

from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors

import data_loading as db

class ContentBasedRecommendationEngine:
    def __init__(self, embeddings_file:str='models/articles_embeddings.pickle',
                 n_recs:int=5,
                 metric:str='cosine'):
        
        available_articles = (
            db.get_all_articles()['article_id']
            .sort_values()
            .to_list()
        )

        embeddings = pd.read_pickle(embeddings_file)
        self.embeddings = embeddings[available_articles]
        # Normalize embeddings once for cosine similarity via dot product
        self.embeddings = normalize(self.embeddings, axis=1)
        self.article_ids = np.array(available_articles)
        self.article_ids_to_index = {aid: idx for idx, aid in enumerate(self.article_ids)}

    
    def recommend(self, article_id: int, n_recs=None):
        if article_id not in self.article_ids:
            return None
        
        article_idx = self.article_ids_to_index[article_id]
        q = self.embeddings[article_idx] # type: ignore
        q = q / np.linalg.norm(q)  # normalize query
        sims = self.embeddings @ q  # assuming embeddings pre-normalized

        # Optional: map cosine [-1,1] → [0,1]
        sims = (sims + 1) / 2

        sorted_idx = np.argsort(-sims)
        sorted_idx = [i for i in sorted_idx if i != article_idx]

        if n_recs is not None:
            top_idx = sorted_idx[:n_recs]
            top_scores = sims[top_idx]
            recs = list(zip(self.article_ids[top_idx], top_scores))
        else:
            recs = list(zip(self.article_ids, sims))

        return recs