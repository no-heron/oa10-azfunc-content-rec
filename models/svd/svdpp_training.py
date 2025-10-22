import pickle
from surprise import Dataset, Reader, SVDpp

import src.data_loading as db

def build_train_model(save_model_path):
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