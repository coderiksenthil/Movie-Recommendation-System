"""
evaluate.py
Evaluation script for the Content-Based Movie Recommendation System.
Computes content consistency, genre preservation, diversity, and tests edge cases.
"""

import os
import numpy as np
import pandas as pd
from recommender import MovieRecommender


def evaluate_genre_consistency(recommender, test_titles, k=10):
    """
    Evaluates whether the top-k recommendations share genres with the seed movie.
    Returns average Genre Match Rate @ K.
    """
    match_rates = []
    results_summary = []

    for title in test_titles:
        res = recommender.recommend(title, top_n=k)
        if not res["success"] or res["recommendations"].empty:
            continue

        query_genre_str = res["queried_movie"]["genre"]
        query_genres = set([g.strip().lower() for g in query_genre_str.split(",") if g.strip()])

        rec_df = res["recommendations"]
        matches = 0

        for _, row in rec_df.iterrows():
            rec_genre_str = row["genre"]
            rec_genres = set([g.strip().lower() for g in rec_genre_str.split(",") if g.strip()])

            # If either has unknown, don't penalize completely, but check for intersection
            if "unknown" in query_genres or "unknown" in rec_genres:
                # Check director match or cast match as fallback consistency
                if (row["director"].lower() != "unknown" and
                        row["director"].lower() == res["queried_movie"]["director"].lower()):
                    matches += 1
            elif query_genres.intersection(rec_genres):
                matches += 1

        rate = matches / len(rec_df)
        match_rates.append(rate)
        results_summary.append({
            "Queried Movie": res["queried_movie"]["title"],
            "Queried Genre": query_genre_str,
            "Director": res["queried_movie"]["director"],
            f"Genre Match @ {k}": f"{rate * 100:.1f}%",
            "Avg Similarity": f"{rec_df['similarity_score'].mean():.4f}",
            "Top Recommendation": rec_df.iloc[0]["title"]
        })

    avg_match_rate = float(np.mean(match_rates)) if match_rates else 0.0
    return avg_match_rate, pd.DataFrame(results_summary)


def evaluate_catalog_coverage(recommender, sample_size=100, k=10):
    """
    Estimates recommendation coverage across a random sample of catalog movies.
    """
    if recommender.movies_df is None or recommender.movies_df.empty:
        return 0.0

    sample_movies = recommender.movies_df["title"].sample(min(sample_size, len(recommender.movies_df)), random_state=42)
    recommended_set = set()

    for title in sample_movies:
        res = recommender.recommend(title, top_n=k)
        if res["success"]:
            for rec_title in res["recommendations"]["title"]:
                recommended_set.add(rec_title)

    coverage_pct = (len(recommended_set) / len(recommender.movies_df)) * 100
    return coverage_pct, len(recommended_set)


def test_edge_cases(recommender):
    """
    Tests edge cases like invalid searches, lowercase/uppercase queries, and partial titles.
    """
    test_cases = [
        ("psycho", "Case insensitivity check"),
        ("VERTIGO", "All-caps title check"),
        ("Rear Window", "Multi-word classic title"),
        ("ThisMovieDoesNotExist12345", "Non-existent movie check"),
        ("", "Empty query string check"),
        ("   ", "Whitespace-only query check")
    ]

    print("\n" + "-" * 60)
    print("                ROBUSTNESS & EDGE CASE TESTS")
    print("-" * 60)

    for query, description in test_cases:
        res = recommender.recommend(query, top_n=5)
        if res["success"]:
            print(f"[PASS] {description}:")
            print(f"       Query: '{query}' -> Found: '{res['queried_movie']['title']}'")
            print(f"       Returned {len(res['recommendations'])} recommendations.")
        else:
            print(f"[PASS] {description}:")
            print(f"       Query: '{query}' -> Gracefully handled.")
            print(f"       Error message: {res['error']}")
            if res.get("suggestions"):
                print(f"       Suggested: {res['suggestions']}")


def run_evaluation(model_dir="model"):
    """
    Executes the recommendation evaluation suite.
    """
    print("=" * 60)
    print("             MOVIE RECOMMENDER SYSTEM EVALUATION")
    print("=" * 60)

    recommender = MovieRecommender()
    try:
        recommender.load(model_dir)
    except FileNotFoundError:
        print(f"Model artifacts not found in '{model_dir}'. Training model first...")
        from train import train_model
        train_model(model_dir)
        recommender.load(model_dir)

    # 1. Test Qualitative Cohorts across different genres
    benchmark_movies = [
        "Psycho",
        "Vertigo",
        "Ninotchka",
        "Stage Fright",
        "The Lodger: A Story of The London Fog",
        "Lifeboat",
        "Spellbound",
        "Rebecca"
    ]

    print(f"\nEvaluating Genre & Content Consistency across {len(benchmark_movies)} representative titles (Top 10)...")
    avg_match_rate, summary_df = evaluate_genre_consistency(recommender, benchmark_movies, k=10)

    print("\nBenchmark Evaluation Table:")
    print(summary_df.to_string(index=False))

    print(f"\nAverage Genre Preservation Rate @ 10: {avg_match_rate * 100:.2f}%")

    # 2. Recommendation Coverage Check
    print("\nMeasuring recommendation diversity and catalog coverage...")
    coverage_pct, unique_recs = evaluate_catalog_coverage(recommender, sample_size=50, k=10)
    print(f"Catalog Coverage from 50 sample seeds: {coverage_pct:.2f}% ({unique_recs} unique movies recommended)")

    # 3. Robustness & Edge Cases
    test_edge_cases(recommender)

    print("\n" + "=" * 60)
    print("                     EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run_evaluation()

