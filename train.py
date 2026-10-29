"""
train.py
Script to load the dataset, preprocess features, train the TF-IDF recommendation model,
and persist the trained artifacts for inference.
"""

import os
import time
from data_loader import load_movies_dataset
from preprocess import preprocess_data
from recommender import MovieRecommender


def train_model(model_dir="model"):
    """
    Executes the complete training workflow:
    1. Loads raw dataset
    2. Preprocesses textual attributes
    3. Fits TF-IDF vectorizer
    4. Saves model artifacts
    5. Validates with a test recommendation
    """
    start_time = time.time()
    print("=" * 60)
    print("           TRAINING MOVIE RECOMMENDATION SYSTEM")
    print("=" * 60)

    # 1. Load data
    print("\n[Step 1/4] Loading dataset...")
    raw_df = load_movies_dataset()
    print(f"Loaded {len(raw_df)} movie records.")

    # 2. Preprocess data
    print("\n[Step 2/4] Preprocessing and engineering features...")
    processed_df = preprocess_data(raw_df)

    # 3. Fit Recommender
    print("\n[Step 3/4] Fitting TF-IDF Vectorizer & Cosine Similarity Matrix...")
    recommender = MovieRecommender()
    recommender.fit(processed_df, max_features=15000)

    # 4. Save artifacts
    print(f"\n[Step 4/4] Saving model artifacts to '{model_dir}'...")
    recommender.save(model_dir=model_dir)

    elapsed = time.time() - start_time
    print(f"\nTraining successfully finished in {elapsed:.2f} seconds!")

    # Verification test
    test_title = "Psycho"
    print(f"\nVerifying model with test query: '{test_title}'...")
    result = recommender.recommend(test_title, top_n=5)
    if result["success"]:
        print(f"Recommendations for '{result['queried_movie']['title']}' ({result['queried_movie']['genre']}):")
        print(result["recommendations"][["title", "year", "genre", "similarity_percent"]])
    else:
        print(f"Test warning: {result['error']}")

    print("\n" + "=" * 60)
    print("Model is ready for evaluation and web application usage.")
    print("=" * 60)


if __name__ == "__main__":
    train_model()
