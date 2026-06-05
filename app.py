"""
Streamlit frontend for the Movie Recommendation System.
Run: streamlit run app.py
Expects the FastAPI backend running at http://localhost:8000
"""

import os

import streamlit as st
import requests
import pandas as pd

# When deployed on HF Spaces, set the API_BASE_URL Space secret to your
# Render service URL, e.g. https://movie-recommender-api.onrender.com
API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background: #0d0d0f; color: #e8e8e8; }

[data-testid="stSidebar"] { background: #111114; border-right: 1px solid #222; }

.hero-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3.8rem;
    letter-spacing: 3px;
    color: #ffffff;
    line-height: 1;
    margin-bottom: 0.2rem;
}
.hero-sub {
    font-size: 0.95rem;
    color: #888;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 2rem;
}
.movie-card {
    background: #18181c;
    border: 1px solid #2a2a30;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    transition: border-color 0.2s;
}
.movie-card:hover { border-color: #e50914; }
.movie-rank {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2rem;
    color: #333;
    min-width: 2.5rem;
    line-height: 1;
}
.movie-title { font-size: 1rem; font-weight: 600; color: #f0f0f0; margin-bottom: 0.2rem; }
.movie-idx   { font-size: 0.72rem; color: #555; margin-top: 0.2rem; }
.metric-pill {
    display: inline-block;
    background: #1e1e24;
    border: 1px solid #2e2e38;
    border-radius: 20px;
    padding: 0.3rem 0.8rem;
    font-size: 0.8rem;
    color: #aaa;
    margin-right: 0.4rem;
    margin-bottom: 0.4rem;
}
.metric-pill span { color: #fff; font-weight: 600; }
.badge-green { color: #2ecc71; font-weight: 600; }
.badge-red   { color: #e50914; font-weight: 600; }
.hist-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.5rem 0;
    border-bottom: 1px solid #1e1e24;
    font-size: 0.88rem;
}
.hist-rating {
    background: #e50914;
    color: white;
    border-radius: 4px;
    padding: 0.1rem 0.5rem;
    font-weight: 600;
    font-size: 0.8rem;
}
.section-header {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.6rem;
    letter-spacing: 2px;
    color: #fff;
    border-bottom: 2px solid #e50914;
    padding-bottom: 0.3rem;
    margin-bottom: 1.2rem;
}
hr { border-color: #1e1e24; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_data(ttl=30)
def api_health():
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        return r.json() if r.ok else None
    except Exception:
        return None


@st.cache_data(ttl=60)
def api_models():
    try:
        r = requests.get(f"{API_BASE}/models", timeout=3)
        return r.json().get("models", []) if r.ok else []
    except Exception:
        return []


def get_recommendations(user_idx, top_k, model, exclude_seen):
    r = requests.post(f"{API_BASE}/recommendations", json={
        "user_idx": user_idx,
        "top_k": top_k,
        "model": model,
        "exclude_seen": exclude_seen,
    }, timeout=10)
    if r.ok:
        return r.json()
    st.error(f"API error {r.status_code}: {r.json().get('detail', r.text)}")
    return None


def get_user_history(user_idx, limit=20):
    r = requests.get(f"{API_BASE}/users/{user_idx}/history", params={"limit": limit}, timeout=5)
    if r.ok:
        return r.json()
    return None


def predict_rating(user_idx, movie_idx, model):
    r = requests.post(f"{API_BASE}/ratings/predict", json={
        "user_idx": user_idx,
        "movie_idx": movie_idx,
        "model": model,
    }, timeout=5)
    if r.ok:
        return r.json()
    st.error(f"API error: {r.json().get('detail', r.text)}")
    return None


def get_evaluate(model, n_users, top_k):
    r = requests.get(f"{API_BASE}/evaluate", params={
        "model": model, "n_users": n_users, "top_k": top_k
    }, timeout=60)
    if r.ok:
        return r.json()
    st.error(f"Evaluation error: {r.json().get('detail', r.text)}")
    return None


def run_ab_test(user_idx, top_k, model_a, model_b, exclude_seen):
    r = requests.post(f"{API_BASE}/ab-test", json={
        "user_idx": user_idx,
        "top_k": top_k,
        "model_a": model_a,
        "model_b": model_b,
        "exclude_seen": exclude_seen,
    }, timeout=15)
    if r.ok:
        return r.json()
    st.error(f"API error: {r.json().get('detail', r.text)}")
    return None


def stars(rating):
    filled = int(round(rating))
    return "★" * filled + "☆" * (5 - filled)


def genre_tags(genres_str):
    tags = genres_str.split("|") if genres_str else []
    return " ".join(
        f'<span style="background:#1e1e24;border:1px solid #333;border-radius:4px;'
        f'padding:1px 6px;font-size:0.72rem;color:#aaa;margin-right:3px">{g}</span>'
        for g in tags[:4]
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div style="font-size:2rem">🎬</div>', unsafe_allow_html=True)
    st.markdown("### Movie Recommender")
    st.markdown(
        '<div style="font-size:0.75rem;color:#666;text-transform:uppercase;'
        'letter-spacing:1px;margin-bottom:1.5rem">Powered by SVD · CF · Two-Tower</div>',
        unsafe_allow_html=True,
    )

    health = api_health()
    if health:
        st.markdown('<div class="badge-green">● API Online</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="badge-red">● API Offline — start your FastAPI server</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    page = st.radio(
        "Navigate",
        ["🎯 Recommendations", "📋 User History", "⭐ Rate Prediction", "📊 Evaluate Models", "🧪 A/B Test"],
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown("**Model Status**")
    models_info = api_models()
    for m in models_info:
        icon = "🟢" if m["trained"] else "🔴"
        loaded = " *(loaded)*" if m["loaded"] else ""
        st.markdown(f"{icon} `{m['name']}`{loaded}")


# ── Pages ─────────────────────────────────────────────────────────────────────

if page == "🎯 Recommendations":
    st.markdown('<div class="hero-title">RECOMMENDATIONS</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Personalised picks powered by machine learning</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
    with col1:
        user_idx = st.number_input("User Index", min_value=0, max_value=29473, value=42, step=1)
    with col2:
        trained_models = [m["name"] for m in models_info if m["trained"]] or ["svd"]
        model = st.selectbox("Model", trained_models, index=0)
    with col3:
        top_k = st.slider("Top K", min_value=5, max_value=50, value=10, step=5)
    with col4:
        exclude_seen = st.checkbox("Exclude seen", value=True)

    if st.button("🎬 Get Recommendations", use_container_width=True, type="primary"):
        with st.spinner("Finding your next favourites..."):
            result = get_recommendations(user_idx, top_k, model, exclude_seen)

        if result:
            recs = result["recommendations"]
            st.markdown(
                f'<div class="section-header">Top {len(recs)} for User #{user_idx}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div style="margin-bottom:1rem">'
                f'<span class="metric-pill">Model <span>{model.upper()}</span></span>'
                f'<span class="metric-pill">Exclude Seen <span>{"Yes" if exclude_seen else "No"}</span></span>'
                f'</div>',
                unsafe_allow_html=True,
            )

            for i, movie in enumerate(recs, 1):
                st.markdown(f"""
                <div class="movie-card">
                    <div class="movie-rank">{i:02d}</div>
                    <div style="flex:1">
                        <div class="movie-title">{movie['title']}</div>
                        <div style="margin-top:4px">{genre_tags(movie['genres'])}</div>
                        <div class="movie-idx">movie_idx: {movie['movie_idx']}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            df = pd.DataFrame(recs)
            st.download_button(
                "⬇ Download as CSV",
                df.to_csv(index=False),
                f"recs_user{user_idx}_{model}.csv",
                "text/csv",
            )


elif page == "📋 User History":
    st.markdown('<div class="hero-title">USER HISTORY</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Movies rated by a user in the training set</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        user_idx = st.number_input("User Index", min_value=0, max_value=29473, value=42, step=1)
    with col2:
        limit = st.selectbox("Show", [10, 20, 50, 100], index=1)

    if st.button("📋 Load History", use_container_width=True, type="primary"):
        with st.spinner("Fetching history..."):
            history = get_user_history(user_idx, limit)

        if history:
            st.markdown(
                f'<div class="section-header">User #{user_idx} — {history["n_ratings"]} ratings</div>',
                unsafe_allow_html=True,
            )

            for row in history["history"]:
                rating = row.get("rating", 0)
                st.markdown(f"""
                <div class="hist-row">
                    <div>
                        <div style="font-weight:500;color:#f0f0f0">{row['title']}</div>
                        <div style="font-size:0.75rem;color:#555">{row.get('genres', '')}</div>
                    </div>
                    <div style="text-align:right">
                        <div class="hist-rating">{rating:.1f} ★</div>
                        <div style="font-size:0.7rem;color:#444;margin-top:2px">idx {row['movie_idx']}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            df = pd.DataFrame(history["history"])
            if not df.empty:
                avg = df["rating"].mean()
                st.divider()
                col1, col2, col3 = st.columns(3)
                col1.metric("Movies Rated", history["n_ratings"])
                col2.metric("Avg Rating", f"{avg:.2f} ★")
                col3.metric(
                    "Genres",
                    df["genres"].str.split("|").explode().nunique() if "genres" in df else "—",
                )


elif page == "⭐ Rate Prediction":
    st.markdown('<div class="hero-title">RATING PREDICTION</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Predict what rating a user would give a specific movie</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        user_idx = st.number_input("User Index", min_value=0, max_value=29473, value=42, step=1)
    with col2:
        movie_idx = st.number_input("Movie Index", min_value=0, max_value=7641, value=150, step=1)
    with col3:
        trained_models = [m["name"] for m in models_info if m["trained"]] or ["svd"]
        model = st.selectbox("Model", trained_models)

    if st.button("⭐ Predict Rating", use_container_width=True, type="primary"):
        with st.spinner("Predicting..."):
            result = predict_rating(user_idx, movie_idx, model)

        if result:
            predicted = result["predicted_rating"]
            st.divider()
            c1, c2, c3 = st.columns(3)
            c1.metric("Predicted Rating", f"{predicted:.2f} / 5.0")
            c2.metric("Stars", stars(predicted))
            c3.metric("Model", model.upper())

            pct = predicted / 5.0
            bar_color = "#e50914" if pct < 0.5 else "#2ecc71"
            st.markdown(f"""
            <div style="margin-top:1.5rem">
                <div style="font-size:0.8rem;color:#666;margin-bottom:0.4rem;
                            text-transform:uppercase;letter-spacing:1px">Confidence bar</div>
                <div style="background:#1e1e24;border-radius:6px;height:10px;overflow:hidden">
                    <div style="background:{bar_color};width:{pct*100:.0f}%;height:100%;
                                border-radius:6px;transition:width 0.5s"></div>
                </div>
                <div style="display:flex;justify-content:space-between;font-size:0.7rem;
                            color:#444;margin-top:4px"><span>0</span><span>5</span></div>
            </div>
            """, unsafe_allow_html=True)

            if predicted >= 4.0:
                interpretation = "🎯 Strong match — this user is likely to enjoy this movie."
            elif predicted >= 3.0:
                interpretation = "👍 Decent match — user may enjoy it."
            else:
                interpretation = "⚠️ Weak match — user is unlikely to rate this highly."

            st.markdown(f"""
            <div style="margin-top:1.5rem;background:#18181c;border:1px solid #2a2a30;
                        border-radius:10px;padding:1rem">
                <div style="font-size:0.75rem;color:#666;text-transform:uppercase;letter-spacing:1px">
                    Interpretation</div>
                <div style="margin-top:0.5rem;color:#ccc">{interpretation}</div>
            </div>
            """, unsafe_allow_html=True)


elif page == "📊 Evaluate Models":
    st.markdown('<div class="hero-title">MODEL EVALUATION</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Live ranking metrics on the held-out test split</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        trained_models = [m["name"] for m in models_info if m["trained"]] or ["svd"]
        model = st.selectbox("Model", trained_models)
    with col2:
        n_users = st.slider("Users to evaluate", 50, 500, 200, step=50)
    with col3:
        top_k = st.slider("Top K", 5, 50, 10, step=5)

    st.info("⏱ Evaluation may take 10–60 seconds depending on the model and number of users.", icon="ℹ️")

    if st.button("📊 Run Evaluation", use_container_width=True, type="primary"):
        with st.spinner(f"Evaluating {model.upper()} on {n_users} users..."):
            result = get_evaluate(model, n_users, top_k)

        if result:
            metrics = result["metrics"][0] if result["metrics"] else {}
            st.markdown(
                f'<div class="section-header">{model.upper()} @ K={top_k} — {result["n_users"]} users</div>',
                unsafe_allow_html=True,
            )

            cols = st.columns(3)
            metric_map = {
                "Precision@K": ("Precision", "Fraction of top-K that are relevant"),
                "Recall@K":    ("Recall",    "Relevant items found in top-K"),
                "NDCG@K":      ("NDCG",      "Normalised Discounted Cumulative Gain"),
                "MAP":         ("MAP",       "Mean Average Precision"),
                "MRR":         ("MRR",       "Mean Reciprocal Rank"),
                "HitRate@K":   ("HitRate",   "Any relevant item in top-K"),
            }
            for i, (key, (label, help_txt)) in enumerate(metric_map.items()):
                if key in metrics:
                    cols[i % 3].metric(label, f"{metrics[key]:.4f}", help=help_txt)

            if "Coverage@K" in metrics:
                st.divider()
                c1, c2 = st.columns(2)
                c1.metric(
                    "Catalogue Coverage",
                    f"{metrics['Coverage@K']:.2%}",
                    help="% of catalogue recommended to ≥1 user",
                )
                if "Novelty@K" in metrics:
                    c2.metric(
                        "Novelty",
                        f"{metrics['Novelty@K']:.2f}",
                        help="Mean self-information — higher = more surprising recommendations",
                    )

            with st.expander("Raw metrics JSON"):
                st.json(metrics)


elif page == "🧪 A/B Test":
    st.markdown('<div class="hero-title">A/B TEST</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Compare two models side by side for the same user</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
    with col1:
        user_idx = st.number_input("User Index", min_value=0, max_value=29473, value=42, step=1)
    with col2:
        trained_models = [m["name"] for m in models_info if m["trained"]] or ["svd", "cf"]
        model_a = st.selectbox("Model A", trained_models, index=0)
    with col3:
        model_b_opts = [m for m in trained_models if m != model_a]
        if model_b_opts:
            model_b = st.selectbox("Model B", model_b_opts, index=0)
        else:
            model_b = st.selectbox("Model B", trained_models, index=0)
    with col4:
        top_k = st.slider("Top K", 5, 20, 10, step=5)

    if st.button("🧪 Run A/B Test", use_container_width=True, type="primary"):
        with st.spinner("Running both models..."):
            result = run_ab_test(user_idx, top_k, model_a, model_b, exclude_seen=True)

        if result:
            overlap_pct = result["overlap_pct"]

            st.divider()
            c1, c2, c3 = st.columns(3)
            c1.metric("Overlap", f"{overlap_pct:.0f}%", help="% of top-K both models agree on")
            c2.metric(f"Unique to {model_a.upper()}", len(result["unique_to_a"]))
            c3.metric(f"Unique to {model_b.upper()}", len(result["unique_to_b"]))

            if overlap_pct < 40:
                overlap_color = "#2ecc71"
                diversity_msg = "🟢 Low overlap — models capture different signals, good for ensembling."
            elif overlap_pct < 70:
                overlap_color = "#f39c12"
                diversity_msg = "🟡 Moderate overlap — models partially agree."
            else:
                overlap_color = "#e50914"
                diversity_msg = "🔴 High overlap — models recommend very similar items."

            st.markdown(f"""
            <div style="background:#18181c;border:1px solid #2a2a30;border-radius:10px;
                        padding:1rem;margin:1rem 0">
                <div style="font-size:0.75rem;color:#666;text-transform:uppercase;
                            letter-spacing:1px;margin-bottom:0.5rem">Diversity Signal</div>
                <div style="background:#111;border-radius:6px;height:8px;overflow:hidden">
                    <div style="background:{overlap_color};width:{overlap_pct}%;
                                height:100%;border-radius:6px"></div>
                </div>
                <div style="font-size:0.8rem;color:#aaa;margin-top:0.5rem">{diversity_msg}</div>
            </div>
            """, unsafe_allow_html=True)

            st.divider()
            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown(f'<div class="section-header">{model_a.upper()}</div>', unsafe_allow_html=True)
                for i, movie in enumerate(result["results_a"], 1):
                    is_overlap = any(m["movie_idx"] == movie["movie_idx"] for m in result["overlap"])
                    border = "border-color: #2ecc71;" if is_overlap else ""
                    checkmark = " ✓" if is_overlap else ""
                    st.markdown(f"""
                    <div class="movie-card" style="{border}">
                        <div class="movie-rank">{i:02d}</div>
                        <div style="flex:1">
                            <div class="movie-title">{movie['title']}{checkmark}</div>
                            <div style="margin-top:4px">{genre_tags(movie['genres'])}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            with col_b:
                st.markdown(f'<div class="section-header">{model_b.upper()}</div>', unsafe_allow_html=True)
                for i, movie in enumerate(result["results_b"], 1):
                    is_overlap = any(m["movie_idx"] == movie["movie_idx"] for m in result["overlap"])
                    border = "border-color: #2ecc71;" if is_overlap else ""
                    checkmark = " ✓" if is_overlap else ""
                    st.markdown(f"""
                    <div class="movie-card" style="{border}">
                        <div class="movie-rank">{i:02d}</div>
                        <div style="flex:1">
                            <div class="movie-title">{movie['title']}{checkmark}</div>
                            <div style="margin-top:4px">{genre_tags(movie['genres'])}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            st.caption("✓ = movie recommended by both models")