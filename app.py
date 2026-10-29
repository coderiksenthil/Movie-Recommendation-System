"""
app.py
Interactive Streamlit Web Application for the Content-Based Movie Recommendation System.
"""

import streamlit as st
import pandas as pd
from recommender import MovieRecommender
from train import train_model


# Page configuration
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for cards and badges
st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .movie-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        transition: transform 0.15s ease-in-out;
    }
    .movie-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.08);
    }
    .score-badge {
        background-color: #DBEAFE;
        color: #1E40AF;
        padding: 3px 8px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .genre-tag {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 2px 7px;
        border-radius: 8px;
        font-size: 0.8rem;
        margin-right: 5px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_or_train_recommender(model_dir="model"):
    """
    Loads recommender model from disk or trains a new one if not present.
    Cached so it only runs once per app session.
    """
    recommender = MovieRecommender()
    try:
        recommender.load(model_dir)
    except Exception:
        with st.spinner("Model artifacts not found. Training model for the first time..."):
            train_model(model_dir)
            recommender.load(model_dir)
    return recommender


def main():
    st.markdown('<div class="main-title">🎬 Movie Recommendation System</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Discover personalized movie recommendations powered by Machine Learning (TF-IDF & Cosine Similarity)</div>', unsafe_allow_html=True)

    recommender = load_or_train_recommender()
    df = recommender.movies_df

    # Sidebar controls
    st.sidebar.header("⚙️ Configuration")
    top_n = st.sidebar.slider("Number of Recommendations:", min_value=5, max_value=10, value=5, step=1)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Dataset Info")
    st.sidebar.info(f"**Total Movies in Catalog:** {len(df):,}\n\n**Algorithm:** Content-Based TF-IDF + Cosine Similarity")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💡 Try Popular Movies")
    sample_queries = [
        "Psycho",
        "Vertigo",
        "Ninotchka",
        "Stage Fright",
        "Rebecca",
        "Lifeboat",
        "The Lodger: A Story of The London Fog"
    ]

    selected_sample = None
    for sample in sample_queries:
        if st.sidebar.button(sample, key=f"btn_{sample}"):
            selected_sample = sample

    # Search options tab
    tab_search, tab_browse = st.tabs(["🔍 Search Movie", "📋 Select from Catalog"])

    user_query = ""

    with tab_search:
        st.write("Type a movie title or keyword below:")
        col_input, col_btn = st.columns([4, 1])
        with col_input:
            search_input = st.text_input(
                "Movie Title",
                value=selected_sample if selected_sample else "",
                placeholder="e.g. Psycho, Vertigo, Ninotchka...",
                label_visibility="collapsed"
            )
        with col_btn:
            search_clicked = st.button("Search", type="primary", use_container_width=True)

        if search_input:
            user_query = search_input.strip()

    with tab_browse:
        st.write("Or pick a movie directly from the alphabetically sorted catalog:")
        all_titles = sorted(df["title"].unique())
        dropdown_selection = st.selectbox(
            "Select a Movie:",
            options=["-- Select a movie --"] + all_titles,
            index=0
        )
        if dropdown_selection != "-- Select a movie --":
            user_query = dropdown_selection

    # Process and display recommendations
    if user_query:
        result = recommender.recommend(user_query, top_n=top_n)

        if not result["success"]:
            st.error(f"❌ {result['error']}")
            if result.get("suggestions"):
                st.warning("Did you mean one of these titles?")
                cols = st.columns(min(len(result["suggestions"]), 5))
                for i, sug in enumerate(result["suggestions"]):
                    with cols[i]:
                        if st.button(sug, key=f"sug_{i}"):
                            st.session_state["rerun_query"] = sug
                            st.rerun()
            else:
                st.info("Tip: Try searching for a director or a partial title like 'Psycho' or 'Window'.")
            return

        movie = result["queried_movie"]
        recs_df = result["recommendations"]

        st.markdown("---")
        st.subheader("🎯 Selected Movie Details")

        with st.container():
            col1, col2, col3, col4 = st.columns([3, 1, 2, 2])
            with col1:
                st.markdown(f"**Title:** {movie['title']}")
                st.markdown(f"**Director:** {movie['director']}")
            with col2:
                st.markdown(f"**Year:** {movie['year']}")
            with col3:
                st.markdown(f"**Genre:** {movie['genre']}")
            with col4:
                st.markdown(f"**Cast:** {movie['cast']}")

        st.markdown("---")
        st.subheader(f"✨ Top {len(recs_df)} Recommended Movies for '{movie['title']}'")

        for idx, row in recs_df.iterrows():
            with st.container():
                st.markdown(f"""
                <div class="movie-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h4 style="margin: 0; color: #1E293B;">#{idx + 1}. {row['title']} ({row['year']})</h4>
                        <span class="score-badge">Match: {row['similarity_percent']}</span>
                    </div>
                    <p style="margin: 6px 0 0 0; color: #475569; font-size: 0.95rem;">
                        <strong>Director:</strong> {row['director']} &nbsp;|&nbsp; 
                        <strong>Genre:</strong> {row['genre']}
                    </p>
                    <p style="margin: 4px 0 0 0; color: #64748B; font-size: 0.85rem;">
                        <strong>Cast:</strong> {row['cast']}
                    </p>
                </div>
                """, unsafe_allow_html=True)

        with st.expander("📊 View Recommendations Data Table"):
            st.dataframe(
                recs_df[["title", "year", "director", "genre", "cast", "similarity_score", "similarity_percent"]],
                use_container_width=True
            )


if __name__ == "__main__":
    main()

