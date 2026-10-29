"""
preprocess.py
Module responsible for data cleaning, text normalization, feature engineering,
and query matching utilities for movie recommendation.
"""

import re
import pandas as pd
from data_loader import load_movies_dataset


def normalize_text(text):
    """
    Normalizes a text string by lowercasing and stripping punctuation.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def create_combined_features(df):
    """
    Combines relevant movie attributes (genre, director, cast, notes, title)
    into a single textual representation for TF-IDF vectorization.
    """
    df = df.copy()

    # Ensure no NaN values
    df["title"] = df["title"].fillna("").astype(str)
    df["director"] = df["director"].fillna("").astype(str)
    df["genre"] = df["genre"].fillna("").astype(str)
    df["cast"] = df["cast"].fillna("").astype(str)
    df["notes"] = df["notes"].fillna("").astype(str)
    df["year"] = df["year"].fillna("").astype(str)

    combined_list = []
    for _, row in df.iterrows():
        # Clean each attribute
        genre_clean = normalize_text(row["genre"])
        director_clean = normalize_text(row["director"])
        cast_clean = normalize_text(row["cast"])
        notes_clean = normalize_text(row["notes"])
        title_clean = normalize_text(row["title"])

        # Give prominent weight to genre and director by repeating them
        features = f"{genre_clean} {genre_clean} {director_clean} {director_clean} {cast_clean} {notes_clean} {title_clean}"
        combined_list.append(features.strip())

    df["combined_features"] = combined_list
    df["normalized_title"] = df["title"].apply(normalize_text)

    return df


def find_movie_by_title(query, df):
    """
    Finds a movie in the DataFrame given a user query.
    Supports:
    1. Exact case-insensitive match
    2. Exact normalized match
    3. Title starts with query
    4. Title contains query substring
    Returns matching row index and matched title, or (None, None) if not found.
    """
    if not query or not isinstance(query, str):
        return None, None

    query_str = query.strip()
    norm_query = normalize_text(query_str)
    if not norm_query:
        return None, None

    # 1. Exact match (case-insensitive)
    exact_idx = df[df["title"].str.lower() == query_str.lower()].index
    if not exact_idx.empty:
        return exact_idx[0], df.loc[exact_idx[0], "title"]

    # 2. Normalized exact match
    norm_idx = df[df["normalized_title"] == norm_query].index
    if not norm_idx.empty:
        return norm_idx[0], df.loc[norm_idx[0], "title"]

    # 3. Starts with query
    prefix_idx = df[df["normalized_title"].str.startswith(norm_query)].index
    if not prefix_idx.empty:
        return prefix_idx[0], df.loc[prefix_idx[0], "title"]

    # 4. Substring match
    sub_idx = df[df["normalized_title"].str.contains(r"\b" + re.escape(norm_query), regex=True)].index
    if not sub_idx.empty:
        return sub_idx[0], df.loc[sub_idx[0], "title"]

    sub_any_idx = df[df["normalized_title"].str.contains(re.escape(norm_query), regex=True)].index
    if not sub_any_idx.empty:
        return sub_any_idx[0], df.loc[sub_any_idx[0], "title"]

    return None, None


def search_matching_titles(query, df, limit=10):
    """
    Returns a list of candidate movie titles matching the search query.
    Useful for autocomplete or search suggestions.
    """
    if not query or not isinstance(query, str):
        return []

    norm_query = normalize_text(query.strip())
    if not norm_query:
        return []

    # Matches starting with query
    prefix_matches = df[df["normalized_title"].str.startswith(norm_query)]["title"].tolist()

    # Matches containing query as whole word or substring
    contains_matches = df[df["normalized_title"].str.contains(re.escape(norm_query), regex=True)]["title"].tolist()

    # Combine preserving order and uniqueness
    results = []
    seen = set()
    for t in prefix_matches + contains_matches:
        if t not in seen:
            seen.add(t)
            results.append(t)
        if len(results) >= limit:
            break

    return results


def preprocess_data(df=None):
    """
    End-to-end preprocessing pipeline.
    """
    if df is None:
        df = load_movies_dataset()
    print("Preprocessing text features...")
    processed_df = create_combined_features(df)
    print("Preprocessing completed.")
    return processed_df


if __name__ == "__main__":
    df = preprocess_data()
    print("\nPreprocessed DataFrame sample:")
    print(df[["title", "genre", "combined_features"]].head(5))

    test_queries = ["Psycho", "vertigo", "ninotchka", "nonexistent movie"]
    print("\nTesting movie searches:")
    for q in test_queries:
        idx, match = find_movie_by_title(q, df)
        print(f"Query '{q}' -> Match: {match} (Index: {idx})")
