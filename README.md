# 🎬 Movie Recommendation System

A simple, modular, and beginner-friendly Movie Recommendation System built with Python, Machine Learning (TF-IDF & Cosine Similarity), and Streamlit. The system utilizes the local dataset provided in `movie.zip` directly without relying on any external APIs or third-party datasets.

---

## 📌 Project Architecture & Files

The project is organized into 6 modular Python files and essential configuration files:

```
Movie-Recommendation-System/
│
├── data/                    # Dataset directory containing raw HTML files (main.html, casts.html)
├── movie.zip                # Original compressed dataset file
│
├── data_loader.py           # Module to locate, extract, and load movie and cast records
├── preprocess.py            # Text normalization, feature engineering, and search matching
├── recommender.py           # Core TF-IDF vectorizer and Cosine Similarity recommender engine
├── train.py                 # Training script that fits the model and saves serialized artifacts
├── evaluate.py              # Evaluation script computing consistency, coverage, and robustness
├── app.py                   # Streamlit web application for interactive recommendations
│
├── requirements.txt         # Project dependencies (pandas, beautifulsoup4, scikit-learn, streamlit)
├── .gitignore               # Git ignore rules for virtual environments, caches, and models
└── README.md                # Project documentation and usage instructions
```

---

## 🧠 How the Recommendation System Works

1. **Data Loading (`data_loader.py`)**:
   - Parses `main.html` and `casts.html` from the dataset.
   - Decodes HTML entities and cleans tags using BeautifulSoup.
   - Extracts movie metadata: **Title**, **Year**, **Director**, **Genre**, **Cast**, and **Notes**.
   - Handles missing values gracefully by imputing `"Unknown"` or clean defaults.

2. **Preprocessing & Feature Engineering (`preprocess.py`)**:
   - Cleans and normalizes text by lowercasing and stripping special punctuation.
   - Combines multiple content attributes (`genre`, `director`, `cast`, `notes`, `title`) into a unified `combined_features` representation.
   - Features like genre and director are emphasized to ensure strong thematic and stylistic alignment.

3. **Content-Based Filtering (`recommender.py`)**:
   - Transforms textual features into numerical feature vectors using **Term Frequency-Inverse Document Frequency (TF-IDF)** with unigrams and bigrams (`ngram_range=(1, 2)`).
   - Computes pairwise **Cosine Similarity** using `linear_kernel` (optimized for L2-normalized TF-IDF matrices).
   - Retrieves the top 5 to 10 most similar movies, excluding the query movie itself.
   - Handles edge cases: invalid movie titles, casing variations, partial matches, and provides search suggestions.

---

## 🚀 Quickstart Guide

### 1. Prerequisites & Environment Setup

Clone or open the repository in your terminal and create a virtual environment:

```bash
# Optional: Create and activate a virtual environment
python -m venv venv

# Windows:
.\venv\Scripts\activate

# macOS / Linux:
source venv/bin/activate
```

### 2. Install Dependencies

Install the required packages using pip:

```bash
python -m pip install -r requirements.txt
```

---

## ⚙️ Running the Project

### Step 1: Train the Model

To parse the dataset, build the TF-IDF feature matrix, and save model artifacts to `model/`, run:

```bash
python train.py
```

*Example Output:*
```
[Step 1/4] Loading dataset...
Loaded 10736 movie records.
[Step 2/4] Preprocessing and engineering features...
[Step 3/4] Fitting TF-IDF Vectorizer & Cosine Similarity Matrix...
TF-IDF matrix built with shape (10736, 15000).
[Step 4/4] Saving model artifacts to 'model'...
Training successfully finished in ~10 seconds!
```

---

### Step 2: Evaluate the Recommendation System

To run the automated evaluation suite (checking genre preservation rate, catalog coverage, and robustness against edge cases), run:

```bash
python evaluate.py
```

*Key Evaluation Metrics:*
- **Genre Match Rate @ 10**: Evaluates how consistently top recommendations match the queried movie's genre/thematic style (achieves >70% on benchmark classic sets).
- **Catalog Coverage**: Verifies that recommendations span a diverse set of movies across the 10,700+ catalog.
- **Robustness Tests**: Validates case-insensitivity, substring matches, empty queries, and nonexistent titles.

---

### Step 3: Launch the Streamlit Web Application

To launch the interactive web user interface in your browser:

```bash
streamlit run app.py
```

Or using python module syntax:

```bash
python -m streamlit run app.py
```

The application will open automatically at `http://localhost:8501`.

---

## 🌟 Web Application Features

- **Search Movie by Name**: Type any title or keyword (e.g. `Psycho`, `Vertigo`, `Ninotchka`, `Rebecca`).
- **Catalog Dropdown**: Browse and select from the full list of over 10,700 movies.
- **Adjustable Top-N Slider**: Choose between **5 to 10** recommendations.
- **Detailed Movie Card Display**: Displays Title, Release Year, Director, Genre, Cast, and Similarity Match Percentage.
- **Graceful Error Handling**: If an invalid title is entered, suggestions and helpful feedback are displayed.
- **Quick-Pick Popular Buttons**: Easily test recommendations with one click from the sidebar.

---

## 🛠️ Technologies Used

- **Python 3.x**
- **Pandas**: Structured tabular data handling and data manipulation.
- **BeautifulSoup4**: Robust HTML parsing and text cleaning from raw dataset files.
- **Scikit-Learn**: Machine learning vectorization (`TfidfVectorizer`) and metric computation (`linear_kernel` / Cosine Similarity).
- **Streamlit**: Fast, responsive web UI for interactive recommendation exploration.
