"""
recommender.py
Module implementing content-based movie recommendations using TF-IDF and Cosine Similarity.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from preprocess import find_movie_by_title, search_matching_titles


class MovieRecommender:
    """
    Content-Based Movie Recommender using TF-IDF feature vectors and Cosine Similarity.
    """

    def __init__(self, movies_df=None, vectorizer=None, tfidf_matrix=None):
        self.movies_df = movies_df
        self.vectorizer = vectorizer
        self.tfidf_matrix = tfidf_matrix

    def fit(self, movies_df, max_features=15000):
        """
        Fits a TF-IDF vectorizer on the combined movie features.
        """
        self.movies_df = movies_df.reset_index(drop=True)
        print("Vectorizing textual features with TF-IDF...")
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=max_features,
            ngram_range=(1, 2)
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.movies_df["combined_features"])
        print(f"TF-IDF matrix built with shape {self.tfidf_matrix.shape}.")
        return self

    def save(self, model_dir="model"):
        """
        Saves the fitted recommender artifacts to disk.
        """
        os.makedirs(model_dir, exist_ok=True)
        with open(os.path.join(model_dir, "movies_df.pkl"), "wb") as f:
            pickle.dump(self.movies_df, f)
        with open(os.path.join(model_dir, "tfidf_vectorizer.pkl"), "wb") as f:
            pickle.dump(self.vectorizer, f)
        with open(os.path.join(model_dir, "tfidf_matrix.pkl"), "wb") as f:
            pickle.dump(self.tfidf_matrix, f)
        print(f"Model artifacts successfully saved to '{model_dir}'.")

    def load(self, model_dir="model"):
        """
        Loads the saved recommender artifacts from disk.
        """
        df_path = os.path.join(model_dir, "movies_df.pkl")
        vec_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")
        mat_path = os.path.join(model_dir, "tfidf_matrix.pkl")

        if not (os.path.exists(df_path) and os.path.exists(vec_path) and os.path.exists(mat_path)):
            raise FileNotFoundError(f"Model artifacts not found in '{model_dir}'. Please train the model first.")

        with open(df_path, "rb") as f:
            self.movies_df = pickle.load(f)
        with open(vec_path, "rb") as f:
            self.vectorizer = pickle.load(f)
        with open(mat_path, "rb") as f:
            self.tfidf_matrix = pickle.load(f)

        print(f"Model loaded successfully from '{model_dir}' ({len(self.movies_df)} movies).")
        return self

    def get_movie_details(self, query):
        """
        Retrieves details for a specific movie if found.
        """
        if self.movies_df is None:
            raise ValueError("Recommender is not trained or loaded yet.")

        idx, matched_title = find_movie_by_title(query, self.movies_df)
        if idx is None:
            return None

        row = self.movies_df.loc[idx]
        return {
            "index": idx,
            "film_id": row.get("film_id", ""),
            "title": row.get("title", ""),
            "year": row.get("year", "Unknown"),
            "director": row.get("director", "Unknown"),
            "genre": row.get("genre", "Unknown"),
            "cast": row.get("cast", "Unknown"),
            "notes": row.get("notes", "")
        }

    def recommend(self, query_title, top_n=10):
        """
        Recommends top_n movies similar to the given title based on cosine similarity.
        Handles missing or invalid searches gracefully.
        """
        if self.movies_df is None or self.tfidf_matrix is None:
            raise ValueError("Recommender model has not been trained or loaded.")

        # Cap top_n between 5 and 20
        top_n = max(1, min(top_n, 20))

        movie_idx, matched_title = find_movie_by_title(query_title, self.movies_df)
        if movie_idx is None:
            suggestions = search_matching_titles(query_title, self.movies_df, limit=5)
            return {
                "success": False,
                "error": f"Movie '{query_title}' was not found in the database.",
                "suggestions": suggestions,
                "queried_movie": None,
                "recommendations": pd.DataFrame()
            }

        # Calculate cosine similarity with all movies using linear_kernel (TF-IDF is L2 normalized)
        query_vec = self.tfidf_matrix[movie_idx]
        sim_scores = linear_kernel(query_vec, self.tfidf_matrix).flatten()

        # Sort indices by similarity descending
        sim_indices = np.argsort(sim_scores)[::-1]

        # Filter out the queried movie itself
        rec_indices = [i for i in sim_indices if i != movie_idx][:top_n]

        queried_details = self.get_movie_details(matched_title)

        recommendations = []
        for i in rec_indices:
            row = self.movies_df.loc[i]
            score = float(sim_scores[i])
            recommendations.append({
                "title": row.get("title", "Unknown"),
                "year": row.get("year", "Unknown"),
                "director": row.get("director", "Unknown"),
                "genre": row.get("genre", "Unknown"),
                "cast": row.get("cast", "Unknown"),
                "similarity_score": round(score, 4),
                "similarity_percent": f"{score * 100:.1f}%"
            })

        rec_df = pd.DataFrame(recommendations)

        return {
            "success": True,
            "error": None,
            "suggestions": [],
            "queried_movie": queried_details,
            "recommendations": rec_df
        }

    def search(self, query, limit=10):
        """
        Searches available movie titles for user suggestions.
        """
        if self.movies_df is None:
            return []
        return search_matching_titles(query, self.movies_df, limit=limit)
