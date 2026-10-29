"""
data_loader.py
Module responsible for locating, extracting, and loading the movie dataset from the provided files.
"""

import os
import zipfile
import pandas as pd
from bs4 import BeautifulSoup


GENRE_CODE_MAP = {
    'actn': 'Action',
    'advt': 'Adventure',
    'biop': 'Biography',
    'cart': 'Cartoon',
    'cnr': 'Cops and Robbers',
    'comd': 'Comedy',
    'docu': 'Documentary',
    'dram': 'Drama',
    'faml': 'Family',
    'fant': 'Fantasy',
    'hist': 'History',
    'horr': 'Horror',
    'musc': 'Musical',
    'myst': 'Mystery',
    'noir': 'Film-Noir',
    'porn': 'Adult',
    'romt': 'Romance',
    'scfi': 'Sci-Fi',
    'susp': 'Suspense',
    'west': 'Western',
    'disa': 'Disaster',
}


def ensure_dataset_extracted(zip_path="movie.zip", target_dir="data"):
    """
    Checks if dataset HTML files exist. If not and zip_path exists, extracts it.
    """
    main_paths = [
        os.path.join(target_dir, "main.html"),
        os.path.join(target_dir, "data", "main.html"),
        "main.html"
    ]
    if any(os.path.exists(p) for p in main_paths):
        return

    # Check zip in current directory or target_dir
    possible_zips = [zip_path, os.path.join(target_dir, zip_path)]
    for z in possible_zips:
        if os.path.exists(z):
            print(f"Extracting dataset from {z}...")
            with zipfile.ZipFile(z, 'r') as zip_ref:
                zip_ref.extractall(target_dir)
            print("Extraction complete.")
            return


def find_dataset_file(filename, search_dirs=None):
    """
    Search for a specific file across standard locations.
    """
    if search_dirs is None:
        search_dirs = [
            "data",
            os.path.join("data", "data"),
            "."
        ]
    for d in search_dirs:
        candidate = os.path.join(d, filename)
        if os.path.exists(candidate):
            return candidate
    return None


def clean_html_text(text):
    """
    Uses BeautifulSoup to decode HTML entities and strip stray HTML tags.
    """
    if not text:
        return ""
    soup = BeautifulSoup(text, "html.parser")
    cleaned = soup.get_text(separator=" ").strip()
    return cleaned


def parse_casts(casts_path):
    """
    Parses casts.html into a dictionary mapping film_id -> list of actor names.
    """
    cast_dict = {}
    if not casts_path or not os.path.exists(casts_path):
        return cast_dict

    with open(casts_path, "r", encoding="latin-1", errors="replace") as f:
        for line in f:
            if "<tr>" in line and "T:" in line:
                parts = [p.replace("</tr>", "").replace("|", "").strip() for p in line.split("<td>")]
                t_indices = [i for i, p in enumerate(parts) if p.startswith("T:")]
                if not t_indices:
                    continue
                t_idx = t_indices[0]
                film_id = parts[t_idx - 1] if t_idx > 0 else ""
                film_id = film_id.replace("<tr>", "").strip()

                if len(parts) > t_idx + 1:
                    actor = parts[t_idx + 1].strip()
                    # Filter out role tokens or empty values
                    if actor and not actor.startswith("\\") and not actor.startswith("R:"):
                        actor = clean_html_text(actor)
                        if actor:
                            cast_dict.setdefault(film_id, []).append(actor)
    return cast_dict


def parse_genre(raw_genre):
    """
    Maps shorthand genre codes to readable genre names.
    """
    if not raw_genre:
        return "Unknown"
    cleaned = clean_html_text(raw_genre).strip().lower()
    cleaned = cleaned.replace("ctxx", "").strip()
    if not cleaned:
        return "Unknown"

    tokens = cleaned.replace("/", ",").replace(";", ",").split(",")
    readable_genres = []
    for token in tokens:
        t = token.strip()
        if not t:
            continue
        mapped = GENRE_CODE_MAP.get(t, t.title())
        if mapped not in readable_genres:
            readable_genres.append(mapped)

    return ", ".join(readable_genres) if readable_genres else "Unknown"


def load_movies_dataset():
    """
    Loads and parses movies and cast information from the dataset files.
    Returns a cleaned pandas DataFrame.
    """
    ensure_dataset_extracted()

    main_path = find_dataset_file("main.html")
    casts_path = find_dataset_file("casts.html")

    if not main_path:
        raise FileNotFoundError("Could not find main.html in dataset directories.")

    print(f"Loading movies from {main_path}...")
    cast_dict = parse_casts(casts_path)
    print(f"Loaded cast data for {len(cast_dict)} movies from {casts_path}...")

    records = []
    with open(main_path, "r", encoding="latin-1", errors="replace") as f:
        for line in f:
            if "<tr>" in line and "T:" in line:
                parts = [p.replace("</tr>", "").replace("|", "").strip() for p in line.split("<td>")]
                t_indices = [i for i, p in enumerate(parts) if p.startswith("T:")]
                if not t_indices:
                    continue
                t_idx = t_indices[0]

                film_id = parts[t_idx - 1] if t_idx > 0 else ""
                film_id = film_id.replace("<tr>", "").strip()
                # Remove any figure prefixes
                if " " in film_id:
                    film_id = film_id.split()[-1]

                raw_title = parts[t_idx][2:].strip()
                title = clean_html_text(raw_title)

                # Skip invalid or empty titles
                if not title or title.lower() in ["", "none", "unknown", "untitled"]:
                    continue

                year = parts[t_idx + 1].strip() if len(parts) > t_idx + 1 else ""
                year = clean_html_text(year)
                # Keep 4-digit year if possible
                year = "".join([c for c in year if c.isdigit()])[:4]
                if not year:
                    year = "Unknown"

                raw_director = parts[t_idx + 2].strip() if len(parts) > t_idx + 2 else ""
                if raw_director.startswith("D:"):
                    raw_director = raw_director[2:].strip()
                director = clean_html_text(raw_director)
                if not director or director.lower() in ["none", ""]:
                    director = "Unknown"

                raw_genre = parts[t_idx + 6].strip() if len(parts) > t_idx + 6 else ""
                genre = parse_genre(raw_genre)

                notes = " ".join(parts[t_idx + 7:]).strip() if len(parts) > t_idx + 7 else ""
                notes = clean_html_text(notes)

                # Fetch cast for this movie
                actors_list = cast_dict.get(film_id, [])
                cast_str = ", ".join(actors_list[:6]) if actors_list else "Unknown"

                records.append({
                    "film_id": film_id,
                    "title": title,
                    "year": year,
                    "director": director,
                    "genre": genre,
                    "cast": cast_str,
                    "notes": notes
                })

    df = pd.DataFrame(records)

    # Drop duplicate movies based on title and year
    df = df.drop_duplicates(subset=["title", "year"]).reset_index(drop=True)

    # Handle missing values
    df["director"] = df["director"].fillna("Unknown")
    df["genre"] = df["genre"].fillna("Unknown")
    df["cast"] = df["cast"].fillna("Unknown")
    df["notes"] = df["notes"].fillna("")

    print(f"Successfully loaded and structured {len(df)} movies.")
    return df


if __name__ == "__main__":
    movies_df = load_movies_dataset()
    print("\nDataset Overview:")
    print(movies_df.info())
    print("\nSample records:")
    print(movies_df[["film_id", "title", "year", "director", "genre", "cast"]].head(10))
