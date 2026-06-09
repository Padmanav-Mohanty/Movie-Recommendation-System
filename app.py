"""
Gradio frontend for the Movie Recommendation System.
Run: python app.py
Expects the FastAPI backend running at http://localhost:8000
(or set API_BASE_URL env var to your deployed API URL)
"""

import os

import gradio as gr
import pandas as pd
import requests

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")


# ── API helpers ───────────────────────────────────────────────────────────────

def api_health():
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        return r.json() if r.ok else None
    except Exception:
        return None


def api_models():
    try:
        r = requests.get(f"{API_BASE}/models", timeout=3)
        return r.json().get("models", []) if r.ok else []
    except Exception:
        return []


def trained_model_names():
    models = api_models()
    names = [m["name"] for m in models if m["trained"]]
    return names or ["svd"]


def get_recommendations(user_idx, top_k, model, exclude_seen):
    r = requests.post(
        f"{API_BASE}/recommendations",
        json={
            "user_idx": int(user_idx),
            "top_k": int(top_k),
            "model": model,
            "exclude_seen": exclude_seen,
        },
        timeout=10,
    )
    if r.ok:
        return r.json()
    return {"error": f"API error {r.status_code}: {r.json().get('detail', r.text)}"}


def get_user_history(user_idx, limit=20):
    r = requests.get(
        f"{API_BASE}/users/{user_idx}/history", params={"limit": limit}, timeout=5
    )
    return r.json() if r.ok else None


def predict_rating(user_idx, movie_idx, model):
    r = requests.post(
        f"{API_BASE}/ratings/predict",
        json={"user_idx": int(user_idx), "movie_idx": int(movie_idx), "model": model},
        timeout=5,
    )
    return r.json() if r.ok else {"error": r.json().get("detail", r.text)}


def get_evaluate(model, n_users, top_k):
    r = requests.get(
        f"{API_BASE}/evaluate",
        params={"model": model, "n_users": n_users, "top_k": top_k},
        timeout=60,
    )
    return r.json() if r.ok else {"error": r.json().get("detail", r.text)}


def run_ab_test(user_idx, top_k, model_a, model_b, exclude_seen):
    r = requests.post(
        f"{API_BASE}/ab-test",
        json={
            "user_idx": int(user_idx),
            "top_k": int(top_k),
            "model_a": model_a,
            "model_b": model_b,
            "exclude_seen": exclude_seen,
        },
        timeout=15,
    )
    return r.json() if r.ok else {"error": r.json().get("detail", r.text)}


# ── Rendering helpers ─────────────────────────────────────────────────────────

def genre_badges(genres_str):
    tags = genres_str.split("|") if genres_str else []
    return " ".join(
        f'<span style="background:#1e1e24;border:1px solid #333;border-radius:4px;'
        f'padding:2px 7px;font-size:0.72rem;color:#aaa;margin-right:3px">{g}</span>'
        for g in tags[:4]
    )


def stars(rating):
    filled = int(round(float(rating)))
    return "★" * filled + "☆" * (5 - filled)


CARD_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;500;600&display=swap');
body, .gradio-container { font-family: 'Inter', sans-serif !important; }
.movie-card {
    background: #18181c; border: 1px solid #2a2a30; border-radius: 10px;
    padding: 0.9rem 1.1rem; margin-bottom: 0.5rem;
    display: flex; align-items: flex-start; gap: 1rem;
    transition: border-color 0.2s;
}
.movie-card:hover { border-color: #e50914; }
.movie-rank { font-family:'Bebas Neue',sans-serif; font-size:1.8rem; color:#333; min-width:2.2rem; line-height:1; }
.movie-title { font-size:0.98rem; font-weight:600; color:#f0f0f0; margin-bottom:3px; }
.movie-meta  { font-size:0.7rem; color:#555; margin-top:3px; }
.section-hdr { font-family:'Bebas Neue',sans-serif; font-size:1.5rem; letter-spacing:2px; color:#fff;
               border-bottom:2px solid #e50914; padding-bottom:3px; margin-bottom:1rem; }
.hist-row { display:flex; justify-content:space-between; align-items:center;
            padding:0.45rem 0; border-bottom:1px solid #1e1e24; font-size:0.87rem; }
.hist-rating { background:#e50914; color:white; border-radius:4px; padding:1px 6px;
               font-weight:600; font-size:0.78rem; }
.metric-box { background:#18181c; border:1px solid #2a2a30; border-radius:10px;
              padding:1rem; text-align:center; flex:1; margin:0 4px; }
.metric-label { font-size:0.72rem; color:#666; text-transform:uppercase; letter-spacing:1px; }
.metric-value { font-size:1.6rem; font-weight:700; color:#fff; margin-top:4px; }
.overlap-bar-wrap { background:#111; border-radius:6px; height:8px; overflow:hidden; margin:8px 0; }
.badge-green { color:#2ecc71; font-weight:600; }
.badge-red   { color:#e50914; font-weight:600; }
</style>
"""


# ── Tab 1 — Recommendations ───────────────────────────────────────────────────

def do_recommendations(user_idx, model, top_k, exclude_seen):
    result = get_recommendations(user_idx, top_k, model, exclude_seen)
    if "error" in result:
        return f"<p style='color:#e50914'>{result['error']}</p>", None

    recs = result["recommendations"]
    html = CARD_CSS
    html += f'<div class="section-hdr">Top {len(recs)} for User #{user_idx}</div>'
    for i, m in enumerate(recs, 1):
        html += f"""
        <div class="movie-card">
            <div class="movie-rank">{i:02d}</div>
            <div style="flex:1">
                <div class="movie-title">{m['title']}</div>
                <div style="margin-top:4px">{genre_badges(m['genres'])}</div>
                <div class="movie-meta">movie_idx: {m['movie_idx']}</div>
            </div>
        </div>"""

    df = pd.DataFrame(recs)
    return html, df


# ── Tab 2 — User History ──────────────────────────────────────────────────────

def do_history(user_idx, limit):
    history = get_user_history(int(user_idx), int(limit))
    if not history:
        return "<p style='color:#e50914'>Could not fetch history — is the API running?</p>", None

    html = CARD_CSS
    html += f'<div class="section-hdr">User #{user_idx} — {history["n_ratings"]} ratings</div>'
    for row in history["history"]:
        rating = row.get("rating", 0)
        html += f"""
        <div class="hist-row">
            <div>
                <div style="font-weight:500;color:#f0f0f0">{row['title']}</div>
                <div style="font-size:0.74rem;color:#555">{row.get('genres','')}</div>
            </div>
            <div style="text-align:right">
                <div class="hist-rating">{rating:.1f} ★</div>
                <div style="font-size:0.7rem;color:#444;margin-top:2px">idx {row['movie_idx']}</div>
            </div>
        </div>"""

    df = pd.DataFrame(history["history"])
    return html, df


# ── Tab 3 — Rating Prediction ─────────────────────────────────────────────────

def do_predict(user_idx, movie_idx, model):
    result = predict_rating(user_idx, movie_idx, model)
    if "error" in result:
        return f"<p style='color:#e50914'>{result['error']}</p>"

    predicted = result["predicted_rating"]
    pct = predicted / 5.0
    bar_color = "#2ecc71" if pct >= 0.6 else "#f39c12" if pct >= 0.4 else "#e50914"

    if predicted >= 4.0:
        interp = "🎯 Strong match — this user is likely to enjoy this movie."
    elif predicted >= 3.0:
        interp = "👍 Decent match — user may enjoy it."
    else:
        interp = "⚠️ Weak match — user is unlikely to rate this highly."

    html = CARD_CSS + f"""
    <div style="display:flex;gap:8px;margin-bottom:1rem">
        <div class="metric-box"><div class="metric-label">Predicted Rating</div>
            <div class="metric-value">{predicted:.2f} / 5.0</div></div>
        <div class="metric-box"><div class="metric-label">Stars</div>
            <div class="metric-value" style="font-size:1.3rem">{stars(predicted)}</div></div>
        <div class="metric-box"><div class="metric-label">Model</div>
            <div class="metric-value" style="font-size:1.1rem">{model.upper()}</div></div>
    </div>
    <div style="margin-bottom:1rem">
        <div style="font-size:0.78rem;color:#666;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px">Score bar</div>
        <div class="overlap-bar-wrap">
            <div style="background:{bar_color};width:{pct*100:.0f}%;height:100%;border-radius:6px"></div>
        </div>
        <div style="display:flex;justify-content:space-between;font-size:0.7rem;color:#444"><span>0</span><span>5</span></div>
    </div>
    <div style="background:#18181c;border:1px solid #2a2a30;border-radius:10px;padding:0.9rem">
        <div style="font-size:0.75rem;color:#666;text-transform:uppercase;letter-spacing:1px">Interpretation</div>
        <div style="margin-top:6px;color:#ccc">{interp}</div>
    </div>"""
    return html


# ── Tab 4 — Evaluate Models ───────────────────────────────────────────────────

def do_evaluate(model, n_users, top_k):
    result = get_evaluate(model, int(n_users), int(top_k))
    if "error" in result:
        return f"<p style='color:#e50914'>{result['error']}</p>", None

    metrics = result["metrics"][0] if result["metrics"] else {}
    metric_map = {
        "Precision@K": "Precision",
        "Recall@K": "Recall",
        "NDCG@K": "NDCG",
        "MAP": "MAP",
        "MRR": "MRR",
        "HitRate@K": "HitRate",
    }

    html = CARD_CSS
    html += f'<div class="section-hdr">{model.upper()} @ K={top_k} — {result["n_users"]} users</div>'
    html += '<div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:1rem">'
    for key, label in metric_map.items():
        if key in metrics:
            html += f"""<div class="metric-box" style="min-width:120px">
                <div class="metric-label">{label}</div>
                <div class="metric-value" style="font-size:1.3rem">{metrics[key]:.4f}</div>
            </div>"""
    html += "</div>"

    if "Coverage@K" in metrics:
        html += f"""<div style="display:flex;gap:8px">
            <div class="metric-box"><div class="metric-label">Coverage</div>
                <div class="metric-value" style="font-size:1.2rem">{metrics['Coverage@K']:.2%}</div></div>"""
        if "Novelty@K" in metrics:
            html += f"""<div class="metric-box"><div class="metric-label">Novelty</div>
                <div class="metric-value" style="font-size:1.2rem">{metrics['Novelty@K']:.2f}</div></div>"""
        html += "</div>"

    df = pd.DataFrame([metrics])
    return html, df


# ── Tab 5 — A/B Test ──────────────────────────────────────────────────────────

def do_ab_test(user_idx, top_k, model_a, model_b, exclude_seen):
    result = run_ab_test(user_idx, top_k, model_a, model_b, exclude_seen)
    if "error" in result:
        return f"<p style='color:#e50914'>{result['error']}</p>"

    overlap_pct = result["overlap_pct"]
    overlap_color = "#2ecc71" if overlap_pct < 40 else "#f39c12" if overlap_pct < 70 else "#e50914"
    if overlap_pct < 40:
        div_msg = "🟢 Low overlap — models capture different signals, good for ensembling."
    elif overlap_pct < 70:
        div_msg = "🟡 Moderate overlap — models partially agree."
    else:
        div_msg = "🔴 High overlap — models recommend very similar items."

    html = CARD_CSS
    html += f"""
    <div style="display:flex;gap:8px;margin-bottom:1rem">
        <div class="metric-box"><div class="metric-label">Overlap</div>
            <div class="metric-value">{overlap_pct:.0f}%</div></div>
        <div class="metric-box"><div class="metric-label">Unique to {model_a.upper()}</div>
            <div class="metric-value">{len(result['unique_to_a'])}</div></div>
        <div class="metric-box"><div class="metric-label">Unique to {model_b.upper()}</div>
            <div class="metric-value">{len(result['unique_to_b'])}</div></div>
    </div>
    <div style="background:#18181c;border:1px solid #2a2a30;border-radius:10px;padding:0.9rem;margin-bottom:1rem">
        <div class="overlap-bar-wrap">
            <div style="background:{overlap_color};width:{overlap_pct}%;height:100%;border-radius:6px"></div>
        </div>
        <div style="font-size:0.82rem;color:#aaa;margin-top:6px">{div_msg}</div>
    </div>
    <div style="display:flex;gap:12px">"""

    for label, movies in [(model_a.upper(), result["results_a"]), (model_b.upper(), result["results_b"])]:
        overlap_ids = {m["movie_idx"] for m in result["overlap"]}
        html += f'<div style="flex:1"><div class="section-hdr">{label}</div>'
        for i, m in enumerate(movies, 1):
            is_shared = m["movie_idx"] in overlap_ids
            border = "border-color:#2ecc71;" if is_shared else ""
            checkmark = " ✓" if is_shared else ""
            html += f"""
            <div class="movie-card" style="{border}">
                <div class="movie-rank">{i:02d}</div>
                <div style="flex:1">
                    <div class="movie-title">{m['title']}{checkmark}</div>
                    <div style="margin-top:4px">{genre_badges(m['genres'])}</div>
                </div>
            </div>"""
        html += "</div>"

    html += "</div><div style='font-size:0.75rem;color:#555;margin-top:8px'>✓ = recommended by both models</div>"
    return html


# ── Status helper for header ──────────────────────────────────────────────────

def api_status_html():
    health = api_health()
    if health:
        return '<span class="badge-green">● API Online</span>'
    return '<span class="badge-red">● API Offline — start your FastAPI server</span>'


# ── Build the Gradio UI ───────────────────────────────────────────────────────

HEADER_HTML = """
<div style="background:#0d0d0f;padding:1.5rem 2rem;border-bottom:1px solid #222;display:flex;align-items:center;justify-content:space-between">
    <div>
        <div style="font-family:'Bebas Neue',sans-serif;font-size:2.8rem;letter-spacing:3px;color:#fff;line-height:1">🎬 MOVIE RECOMMENDER</div>
        <div style="font-size:0.8rem;color:#666;text-transform:uppercase;letter-spacing:2px;margin-top:2px">Powered by SVD · Two-Tower</div>
    </div>
</div>
"""

theme = gr.themes.Base(
    primary_hue="red",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
).set(
    body_background_fill="#0d0d0f",
    body_text_color="#e8e8e8",
    block_background_fill="#18181c",
    block_border_color="#2a2a30",
    input_background_fill="#111114",
    button_primary_background_fill="#e50914",
    button_primary_background_fill_hover="#c0070f",
    button_primary_text_color="#ffffff",
)

with gr.Blocks(theme=theme, title="Movie Recommender", css="""
    .gradio-container { max-width: 1100px !important; }
    footer { display: none !important; }
""") as demo:

    gr.HTML(HEADER_HTML + CARD_CSS)

    with gr.Tabs():

        # ── Tab 1: Recommendations ───────────────────────────────────────────
        with gr.Tab("🎯 Recommendations"):
            gr.Markdown("### Personalised picks powered by machine learning")
            with gr.Row():
                r_user = gr.Number(label="User Index", value=42, minimum=0, maximum=29473, step=1, scale=2)
                r_model = gr.Dropdown(label="Model", choices=trained_model_names(), value="svd", scale=2)
                r_topk = gr.Slider(label="Top K", minimum=5, maximum=50, value=10, step=5, scale=2)
                r_excl = gr.Checkbox(label="Exclude Seen", value=True, scale=1)
            r_btn = gr.Button("🎬 Get Recommendations", variant="primary")
            r_html = gr.HTML()
            r_csv  = gr.Dataframe(label="Results (downloadable)", visible=True, interactive=False)
            r_btn.click(do_recommendations, inputs=[r_user, r_model, r_topk, r_excl], outputs=[r_html, r_csv])

        # ── Tab 2: User History ──────────────────────────────────────────────
        with gr.Tab("📋 User History"):
            gr.Markdown("### Movies rated by a user in the training set")
            with gr.Row():
                h_user  = gr.Number(label="User Index", value=42, minimum=0, maximum=29473, step=1, scale=3)
                h_limit = gr.Dropdown(label="Show", choices=[10, 20, 50, 100], value=20, scale=1)
            h_btn  = gr.Button("📋 Load History", variant="primary")
            h_html = gr.HTML()
            h_df   = gr.Dataframe(label="History Table", visible=True, interactive=False)
            h_btn.click(do_history, inputs=[h_user, h_limit], outputs=[h_html, h_df])

        # ── Tab 3: Rating Prediction ─────────────────────────────────────────
        with gr.Tab("⭐ Rating Prediction"):
            gr.Markdown("### Predict what rating a user would give a specific movie")
            with gr.Row():
                p_user  = gr.Number(label="User Index",  value=42,  minimum=0, maximum=29473, step=1)
                p_movie = gr.Number(label="Movie Index", value=150, minimum=0, maximum=7641, step=1)
                p_model = gr.Dropdown(label="Model", choices=trained_model_names(), value="svd")
            p_btn  = gr.Button("⭐ Predict Rating", variant="primary")
            p_html = gr.HTML()
            p_btn.click(do_predict, inputs=[p_user, p_movie, p_model], outputs=[p_html])

        # ── Tab 4: Evaluate Models ───────────────────────────────────────────
        with gr.Tab("📊 Evaluate Models"):
            gr.Markdown("### Live ranking metrics on the held-out test split")
            gr.Markdown("⏱ Evaluation may take 10–60 s depending on the model and number of users.")
            with gr.Row():
                e_model  = gr.Dropdown(label="Model", choices=trained_model_names(), value="svd")
                e_nusers = gr.Slider(label="Users to evaluate", minimum=50, maximum=500, value=200, step=50)
                e_topk   = gr.Slider(label="Top K", minimum=5, maximum=50, value=10, step=5)
            e_btn  = gr.Button("📊 Run Evaluation", variant="primary")
            e_html = gr.HTML()
            e_df   = gr.Dataframe(label="Metrics Table", visible=True, interactive=False)
            e_btn.click(do_evaluate, inputs=[e_model, e_nusers, e_topk], outputs=[e_html, e_df])

        # ── Tab 5: A/B Test ──────────────────────────────────────────────────
        with gr.Tab("🧪 A/B Test"):
            gr.Markdown("### Compare two models side by side for the same user")
            with gr.Row():
                ab_user = gr.Number(label="User Index", value=42, minimum=0, maximum=29473, step=1, scale=2)
                ab_ma   = gr.Dropdown(label="Model A", choices=trained_model_names(), value="svd", scale=2)
                ab_mb   = gr.Dropdown(label="Model B", choices=trained_model_names(), value="two_tower", scale=2)
                ab_topk = gr.Slider(label="Top K", minimum=5, maximum=20, value=10, step=5, scale=2)
                ab_excl = gr.Checkbox(label="Exclude Seen", value=True, scale=1)
            ab_btn  = gr.Button("🧪 Run A/B Test", variant="primary")
            ab_html = gr.HTML()
            ab_btn.click(do_ab_test, inputs=[ab_user, ab_topk, ab_ma, ab_mb, ab_excl], outputs=[ab_html])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)