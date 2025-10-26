
import os

import logging
import pickle

import pandas as pd
from surprise import Dataset, Reader, SVDpp


from azure_helpers.blob_utils import upload_file_to_blob
from azure_helpers.data_loading import get_interactions, get_user_article_affinity_ratings


# -------------------------------------------------------------------------
# Model Training and Persistence
# -------------------------------------------------------------------------
def __load_training_data(directory = None, file='dataset/clicks_sample.csv'):
    if directory:
        clicks_df = pd.DataFrame()
        for file in os.listdir(directory):
            clicks_df = pd.concat([clicks_df, pd.read_csv(directory + file)])

        # print("\ndTypes when loading from CSV:\n", clicks_df.dtypes)

        clicks_df = clicks_df.drop_duplicates()
    elif file:
        clicks_df = pd.read_csv(file).drop_duplicates()
    else:
        return None
    return clicks_df


def build_and_train_model(save_model_path: str):
    """
    Train a new SVD++ model from user-article affinity ratings
    and persist it to disk.
    """
    try:
        clicks = __load_training_data(file='dataset/clicks_sample.csv')
        interactions = get_interactions(clicks_df=clicks)
        ratings_df = get_user_article_affinity_ratings(interactions_df=interactions)

        if ratings_df.empty:
            raise ValueError("No training data available for SVD++ model.")
        
        # print("\ndTypes at training start:\n", ratings_df.dtypes)

        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(ratings_df, reader)
        trainset = data.build_full_trainset()

        model = SVDpp(
            n_factors=100,
            n_epochs=20,
            lr_all=0.004,
            reg_all=0.04,
            random_state=42
        )
        model.fit(trainset)
        logging.info("Trained SVD++ model on %d interactions.", trainset.n_ratings)

        artifact = {
            "model": model,
            "trainset": trainset,
            "user_raw_to_inner": trainset._raw2inner_id_users,
            "item_raw_to_inner": trainset._raw2inner_id_items,
            "user_inner_to_raw": trainset._inner2raw_id_users,
            "item_inner_to_raw": trainset._inner2raw_id_items
        }

        os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
        with open(save_model_path, "wb") as f:
            pickle.dump(artifact, f)

        upload_file_to_blob(local_path=save_model_path, blob_name='svdpp_model.pkl')
        logging.info("Saved SVD++ model artifact to %s", save_model_path)
        return model, trainset

    except Exception as e:
        logging.exception("Error training SVD++ model: %s", e)
        raise