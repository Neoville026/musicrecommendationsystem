import streamlit as st
import numpy as np
import pandas as pd
import cv2
import json
import os
import pickle
import ast
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import streamlit.components.v1 as components

# --- TensorFlow Import (Opsional - untuk deteksi emosi berbasis kamera) ---
try:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMOTION_MODEL_DIR = os.path.join(BASE_DIR, "model", "model deteksi emosi")
RECOMMENDER_MODEL_DIR = os.path.join(BASE_DIR, "model", "model rekomendasi musik")
DATA_DIR = os.path.join(BASE_DIR, "data")

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
EMOTION_EMOJI = {
    "angry": "😠", "disgust": "🤢", "fear": "😨",
    "happy": "😄", "sad": "😢", "surprise": "😲", "neutral": "😐",
}
TARGET_GENRES = [
    "Dance Pop", "Electronic", "Electropop", "Hip Hop", "Jazz",
    "K-pop", "Latin", "Pop", "Pop Rap", "R&B", "Rock",
]
FEATURE_COLS = [
    "valence", "energy", "danceability", "tempo",
    "acousticness", "instrumentalness", "liveness", "speechiness",
]
IMG_SIZE = 48
N_DISPLAY_CARDS = 10

CARD_GRADIENTS = [
    "linear-gradient(135deg, #e74c3c 0%, #c0392b 100%)",
    "linear-gradient(135deg, #6b7b8d 0%, #8e9eab 100%)",
    "linear-gradient(135deg, #2980b9 0%, #1a5276 100%)",
    "linear-gradient(135deg, #8e44ad 0%, #6c3483 100%)",
    "linear-gradient(135deg, #1db954 0%, #1aa34a 100%)",
    "linear-gradient(135deg, #e67e22 0%, #ca6f1e 100%)",
    "linear-gradient(135deg, #16a085 0%, #117a65 100%)",
    "linear-gradient(135deg, #2c3e50 0%, #1c2833 100%)",
    "linear-gradient(135deg, #d35400 0%, #a04000 100%)",
    "linear-gradient(135deg, #1f618d 0%, #154360 100%)",
]

st.set_page_config(
    page_title="Rekomendasi Musik Deteksi Emosi",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Outfit:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Outfit', sans-serif !important;
}

/* Header Utama */
.main-header {
    padding: 15px 0 30px 0;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 30px;
}
.main-header h1 {
    font-weight: 800 !important;
    font-size: 2.6rem !important;
    background: linear-gradient(135deg, #1DB954, #1ed760);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 10px !important;
    letter-spacing: -0.5px;
}
.main-header p {
    color: rgba(255,255,255,0.65);
    font-size: 1.05rem;
    line-height: 1.7;
    max-width: 950px;
}

/* Kontainer Kaca (Glassmorphism) */
.glass-container {
    background: rgba(22, 29, 48, 0.45);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 24px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
}

.camera-section {
    background: rgba(29, 185, 84, 0.03);
    border: 1px solid rgba(29, 185, 84, 0.15);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 24px;
}
.camera-section h3 {
    color: #1DB954;
    font-weight: 700;
    margin-bottom: 8px;
    font-size: 1.35rem;
}

/* Tampilan Hasil Deteksi */
.detection-result {
    background: rgba(29, 185, 84, 0.06);
    border: 1px solid rgba(29, 185, 84, 0.2);
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    margin-bottom: 20px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.15);
}
.detection-result .emoji {
    font-size: 3.2rem;
    display: block;
    margin-bottom: 8px;
}
.detection-result .label {
    color: #1DB954;
    font-size: 1.6rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 1.5px;
}
.detection-result .conf {
    color: rgba(255,255,255,0.6);
    font-size: 0.92rem;
    margin-top: 6px;
}

/* Bar Distribusi Probabilitas */
.emo-bar-wrap { 
    margin: 8px 0;
    display: flex;
    align-items: center;
}
.emo-bar-label {
    width: 90px;
    font-size: 0.85rem;
    color: rgba(255,255,255,0.8);
    text-align: left;
}
.emo-bar-track {
    flex-grow: 1;
    height: 10px;
    background: rgba(255,255,255,0.06);
    border-radius: 5px;
    overflow: hidden;
    margin: 0 12px;
}
.emo-bar-fill {
    height: 100%;
    border-radius: 5px;
    transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
}
.emo-bar-pct {
    width: 48px;
    font-size: 0.82rem;
    color: rgba(255,255,255,0.5);
    text-align: right;
}

/* Pemetaan Audio Profile */
.profile-map {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 16px 20px;
    margin-top: 15px;
}
.profile-map h4 {
    color: #1DB954;
    font-size: 0.95rem;
    font-weight: 600;
    margin-bottom: 12px;
}
.profile-row {
    display: flex;
    justify-content: space-between;
    padding: 6px 0;
    font-size: 0.86rem;
    color: rgba(255,255,255,0.65);
    border-bottom: 1px solid rgba(255,255,255,0.04);
}
.profile-row:last-child { border-bottom: none; }
.profile-val { color: #1DB954; font-weight: 600; }

.sync-arrow {
    text-align: center;
    font-size: 1.5rem;
    color: #1DB954;
    padding: 10px 0;
    animation: bounce 2s infinite;
}
@keyframes bounce {
    0%, 100% { transform: translateY(0); }
    50%      { transform: translateY(-5px); }
}

.results-header {
    text-align: center;
    padding: 24px 0 12px 0;
}
.results-header h2 {
    color: #1DB954 !important;
    font-weight: 700;
    font-size: 1.85rem;
    letter-spacing: -0.3px;
}

.section-header {
    font-weight: 600;
    font-size: 1.05rem;
    color: rgba(255,255,255,0.9);
    margin-bottom: 12px;
    letter-spacing: -0.1px;
}

.emotion-badge {
    background: rgba(29,185,84,0.1);
    border: 1px solid rgba(29,185,84,0.3);
    border-radius: 12px;
    padding: 12px 18px;
    text-align: center;
    margin-top: 14px;
}
.emotion-badge span {
    color: #1DB954;
    font-weight: 600;
    font-size: 1.05rem;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)

def _build_cbam_model(input_shape=(48, 48, 1), num_classes=7):
    """Membangun arsitektur CNN+CBAM yang sesuai dengan H5."""
    from tensorflow.keras.layers import (
        Input, Conv2D, BatchNormalization, Activation, MaxPooling2D,
        Dense, Dropout, GlobalAveragePooling2D, GaussianNoise,
        Lambda, Multiply, Concatenate, Reshape, Add,
    )
    K = tf.keras.backend

    def _residual_cbam_block(inp, filters, ratio, dropout_rate):
        # 1. Main Path Convolutions
        conv_main1 = Conv2D(filters, (3, 3), padding="same", kernel_initializer="he_normal")(inp)
        conv_main1_bn = BatchNormalization()(conv_main1)
        conv_main1_act = Activation("silu")(conv_main1_bn)
        
        conv_main2 = Conv2D(filters, (3, 3), padding="same", kernel_initializer="he_normal")(conv_main1_act)

        # 2. Residual Shortcut (1x1)
        conv_shortcut = Conv2D(filters, (1, 1), padding="same", kernel_initializer="he_normal")(inp)

        # 3. Batch Normalization (Urutan disinkronkan secara topologis dengan H5)
        conv_main2_bn = BatchNormalization()(conv_main2)
        conv_shortcut_bn = BatchNormalization()(conv_shortcut)

        # 4. Shortcut Connection
        x = Add()([conv_shortcut_bn, conv_main2_bn])
        x = Activation("silu")(x)

        # 5. Squeeze-and-Excitation (Channel Attention)
        se = GlobalAveragePooling2D()(x)
        se = Reshape((1, 1, filters))(se)
        se = Dense(filters // ratio, activation="relu", use_bias=False)(se)
        se = Dense(filters, activation="sigmoid", use_bias=False)(se)
        x = Multiply()([x, se])

        # 6. Spatial Attention
        avg_sp = Lambda(lambda t: K.mean(t, axis=-1, keepdims=True))(x)
        max_sp = Lambda(lambda t: K.max(t, axis=-1, keepdims=True))(x)
        concat = Concatenate(axis=-1)([avg_sp, max_sp])
        sp_map = Conv2D(1, (7, 7), padding="same", activation="sigmoid",
                        kernel_initializer="he_normal", use_bias=False)(concat)
        x = Multiply()([x, sp_map])

        # 7. Pooling + Dropout
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(dropout_rate)(x)
        return x

    inputs = Input(shape=input_shape)
    x = GaussianNoise(0.1)(inputs)

    x = _residual_cbam_block(x, filters=64,  ratio=8, dropout_rate=0.25)
    x = _residual_cbam_block(x, filters=128, ratio=8, dropout_rate=0.30)
    x = _residual_cbam_block(x, filters=256, ratio=8, dropout_rate=0.40)
    x = _residual_cbam_block(x, filters=512, ratio=8, dropout_rate=0.45)

    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="silu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(num_classes, activation="softmax")(x)

    return tf.keras.Model(inputs, x, name="emotion_cnn_cbam")


@st.cache_resource(show_spinner="Memuat model deteksi emosi wajah...")
def load_emotion_model():
    """Memuat model CNN+CBAM secara aman dengan memuat bobot per layer dari H5."""
    if not TF_AVAILABLE:
        return None, None
    try:
        model_path = os.path.join(EMOTION_MODEL_DIR, "emotion_model_final.keras")
        meta_path = os.path.join(EMOTION_MODEL_DIR, "emotion_model_metadata.json")
        if not os.path.exists(model_path):
            return None, None

        metadata = {}
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as fh:
                metadata = json.load(fh)

        num_classes = len(metadata.get("emotions", EMOTIONS))
        model = _build_cbam_model(input_shape=(IMG_SIZE, IMG_SIZE, 1), num_classes=num_classes)

        # Muat bobot secara manual dari zip.keras/model.weights.h5 untuk menghindari bug Keras 3
        import zipfile, tempfile, h5py, numpy as np
        with zipfile.ZipFile(model_path, "r") as zf:
            weight_name = next(name for name in zf.namelist() if name.endswith(".h5"))
            with tempfile.TemporaryDirectory() as tmpdir:
                extracted = zf.extract(weight_name, tmpdir)
                with h5py.File(extracted, "r") as f:
                    layers_grp = f["layers"]
                    for layer in model.layers:
                        if not layer.weights:
                            continue
                        lname = layer.name
                        if lname not in layers_grp:
                            continue
                        vars_grp = layers_grp[lname].get("vars")
                        if vars_grp is None:
                            continue
                        for i, w in enumerate(layer.weights):
                            key = str(i)
                            if key in vars_grp:
                                saved = np.array(vars_grp[key])
                                if saved.shape == tuple(w.shape):
                                    w.assign(saved)
        return model, metadata
    except Exception as exc:
        st.warning(f"Gagal memuat model emosi: {exc}")
        return None, None


@st.cache_resource(show_spinner="Memuat model rekomendasi musik...")
def load_recommender():
    """Memuat model KNN, Scaler, dan metadata rekomendasi."""
    knn_path = os.path.join(RECOMMENDER_MODEL_DIR, "knn_model.pkl")
    scaler_path = os.path.join(RECOMMENDER_MODEL_DIR, "scaler.pkl")
    meta_path = os.path.join(RECOMMENDER_MODEL_DIR, "recommender_metadata.json")

    def _load_pkl(path):
        try:
            import joblib
            return joblib.load(path)
        except Exception:
            with open(path, "rb") as fh:
                return pickle.load(fh)

    try:
        knn = _load_pkl(knn_path)
        scaler = _load_pkl(scaler_path)
        with open(meta_path, "r", encoding="utf-8") as fh:
            metadata = json.load(fh)
        return knn, scaler, metadata
    except Exception as exc:
        st.error(f"Gagal memuat model rekomendasi: {exc}")
        return None, None, None


@st.cache_data(show_spinner="Memuat database musik...")
def load_track_database():
    """Memuat dataset track musik Spotify."""
    try:
        csv_path = os.path.join(RECOMMENDER_MODEL_DIR, "track_db.csv")
        df = pd.read_csv(csv_path)

        def _safe_genres(val):
            try:
                if pd.isna(val):
                    return []
                parsed = ast.literal_eval(str(val))
                return parsed if isinstance(parsed, list) else [str(parsed)]
            except Exception:
                return [str(val)] if pd.notna(val) else []

        df["genres_list"] = df["genres"].apply(_safe_genres)

        year_path = os.path.join(DATA_DIR, "filtered_track_df.csv")
        if os.path.exists(year_path):
            try:
                year_df = pd.read_csv(
                    year_path,
                    usecols=lambda c: c in ("id", "release_year"),
                    on_bad_lines="skip",
                    engine="python",
                )
                if "release_year" in year_df.columns and "id" in year_df.columns:
                    year_df = year_df.drop_duplicates(subset=["id"])
                    df = df.merge(year_df, on="id", how="left")
            except Exception:
                pass
        return df
    except Exception as exc:
        st.error(f"Gagal memuat database musik: {exc}")
        return pd.DataFrame()

# DETEKSI EMOSI (FACIAL DETECTION + MODEL PREDICTION)

def detect_emotion_from_image(image_array, model, emotions_list):
    """Mendeteksi wajah dan mengklasifikasikan emosi menggunakan model CNN+CBAM."""
    if model is None:
        return None, 0.0, None, None

    try:
        if len(image_array.shape) == 3 and image_array.shape[2] >= 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        elif len(image_array.shape) == 2:
            gray = image_array
        else:
            return None, 0.0, None, None

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        if len(faces) == 0:
            return None, 0.0, None, None

        x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
        face_roi = gray[y : y + h, x : x + w]

        face_resized = cv2.resize(face_roi, (IMG_SIZE, IMG_SIZE))
        face_norm = face_resized.astype(np.float32) / 255.0
        face_input = face_norm.reshape(1, IMG_SIZE, IMG_SIZE, 1)

        preds = model.predict(face_input, verbose=0)
        all_probs = preds[0]
        idx = int(np.argmax(all_probs))
        confidence = float(all_probs[idx])
        label = emotions_list[idx]

        annotated = image_array.copy()
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (29, 185, 84), 3)
        cv2.putText(
            annotated,
            f"{label} ({confidence:.0%})",
            (x, y - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (29, 185, 84),
            2,
        )
        return label, confidence, all_probs, annotated
    except Exception:
        return None, 0.0, None, None

# MESIN REKOMENDASI (COSINE SIMILARITY)

def recommend_songs(
    emotion,
    df,
    scaler,
    metadata,
    genre=None,
    n_recs=10,
    year_range=None,
    **feature_overrides,
):
    """Mencari rekomendasi musik terdekat berdasarkan profil emosi dan audio features."""
    feature_cols = metadata["feature_cols"]
    emotion_profiles = metadata["emotion_profiles"]

    if emotion not in emotion_profiles:
        emotion = "neutral"

    profile = emotion_profiles[emotion].copy()

    # Denormalisasi tempo ke skala asli BPM
    tempo_idx = feature_cols.index("tempo")
    t_min = float(scaler.data_min_[tempo_idx])
    t_max = float(scaler.data_max_[tempo_idx])
    profile["tempo"] = profile["tempo"] * (t_max - t_min) + t_min

    # Gunakan modifikasi nilai slider jika ada
    for key, val in feature_overrides.items():
        if key in profile and val is not None:
            profile[key] = val

    query_raw = np.array([[profile[col] for col in feature_cols]])
    query_scaled = scaler.transform(query_raw)

    track_raw = df[feature_cols].values
    track_scaled = scaler.transform(track_raw)

    mask = np.ones(len(df), dtype=bool)
    if genre and genre != "All":
        mask &= df["genres"].str.contains(genre, case=False, na=False).values
    if year_range is not None and "release_year" in df.columns:
        yr = df["release_year"]
        mask &= (yr >= year_range[0]).fillna(True).values & (yr <= year_range[1]).fillna(True).values

    filtered_idx = np.where(mask)[0]
    if len(filtered_idx) == 0:
        return pd.DataFrame()

    filtered_scaled = track_scaled[filtered_idx]
    sims = cosine_similarity(query_scaled, filtered_scaled)[0]

    filtered_df = df.iloc[filtered_idx].copy()
    pop = filtered_df["popularity"].values.astype(float)
    pop_norm = (pop - pop.min()) / (pop.max() - pop.min() + 1e-9)
    scores = sims * 0.7 + pop_norm * 0.3

    n_results = min(n_recs, len(filtered_df))
    top = scores.argsort()[::-1][:n_results]

    result = filtered_df.iloc[top].copy()
    result["similarity"] = sims[top]
    result["final_score"] = scores[top]
    return result.reset_index(drop=True)

# SPOTIFY CARD - RENDER UI

_SPOTIFY_SVG = (
    '<svg viewBox="0 0 24 24" fill="#1DB954">'
    '<path d="M12 0C5.4 0 0 5.4 0 12s5.4 12 12 12 12-5.4 '
    "12-12S18.66 0 12 0zm5.521 17.34c-.24.359-.66.48-1.021"
    ".24-2.82-1.74-6.36-2.101-10.561-1.141-.418.122-.779-"
    ".179-.899-.539-.12-.421.18-.78.54-.9 4.56-1.021 8.52-"
    ".6 11.64 1.32.42.18.479.659.301 1.02zm1.44-3.3c-.301"
    ".42-.841.6-1.262.3-3.239-1.98-8.159-2.58-11.939-1.38-"
    ".479.12-1.02-.12-1.14-.6-.12-.48.12-1.021.6-1.141C9.6"
    " 9.9 15 10.561 18.72 12.84c.361.181.54.78.241 1.2zm"
    ".12-3.36C15.24 8.4 8.82 8.16 5.16 9.301c-.6.179-1.2-"
    ".181-1.38-.721-.18-.601.18-1.2.72-1.381 4.26-1.26 "
    "11.28-1.02 15.721 1.621.539.3.719 1.02.419 1.56-.299"
    '.421-1.02.599-1.559.3z"/></svg>'
)

def render_song_card(track_id, name, artist, similarity, card_idx):
    """Merender kartu lagu Spotify modern dengan iframe compact player."""
    gradient = CARD_GRADIENTS[card_idx % len(CARD_GRADIENTS)]
    spotify_url = f"https://open.spotify.com/track/{track_id}"
    embed_url = f"https://open.spotify.com/embed/track/{track_id}?utm_source=generator&theme=0"

    disp_name = (str(name)[:38] + "...") if len(str(name)) > 38 else str(name)
    disp_artist = (str(artist)[:34] + "...") if len(str(artist)) > 34 else str(artist)
    match_pct = f"{similarity:.0%}"

    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="utf-8"/>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
    *{{margin:0;padding:0;box-sizing:border-box}}
    body{{background:transparent;font-family:'Inter',sans-serif;overflow:hidden}}
    .card{{
        background:{gradient};
        border-radius:16px;padding:18px;position:relative;
        box-shadow:0 8px 32px rgba(0,0,0,.35);
        transition:transform .3s ease,box-shadow .3s ease;
    }}
    .card:hover{{transform:translateY(-4px);box-shadow:0 14px 44px rgba(0,0,0,.55)}}
    .sp-logo{{position:absolute;top:14px;right:14px;width:28px;height:28px;
              opacity:.85;transition:opacity .2s;z-index:5}}
    .sp-logo:hover{{opacity:1}}
    .embed{{border-radius:12px;overflow:hidden;margin:8px 0 12px}}
    .embed iframe{{border-radius:12px;display:block}}
    .title{{color:#fff;font-weight:700;font-size:1.12em;margin:6px 0 5px;
            text-shadow:0 1px 3px rgba(0,0,0,.3)}}
    .artist{{color:rgba(255,255,255,.85);font-size:.88em;margin-bottom:10px}}
    .badge{{background:rgba(0,0,0,.45);padding:2px 8px;border-radius:4px;
            font-size:.68em;font-weight:600;margin-right:6px;letter-spacing:.4px}}
    .foot{{display:flex;align-items:center;justify-content:space-between;margin-top:6px}}
    .save{{color:#1DB954;text-decoration:none;font-size:.86em;font-weight:500;
           transition:color .2s}}
    .save:hover{{color:#1ed760;text-decoration:underline}}
    .match{{color:rgba(255,255,255,.5);font-size:.76em}}
    </style>
    </head>
    <body>
    <div class="card">
      <a class="sp-logo" href="{spotify_url}" target="_blank" title="Buka di Spotify">
        {_SPOTIFY_SVG}
      </a>
      <div class="embed">
        <iframe src="{embed_url}" width="100%" height="152" frameBorder="0"
                allow="autoplay;clipboard-write;encrypted-media;fullscreen;picture-in-picture"
                loading="lazy"></iframe>
      </div>
      <div class="title">{disp_name}</div>
      <div class="artist"><span class="badge">PREVIEW</span>{disp_artist}</div>
      <div class="foot">
        <a class="save" href="{spotify_url}" target="_blank">+ Simpan di Spotify</a>
      </div>
    </div>
    </body>
    </html>
    """
    components.html(html, height=355, scrolling=False)


def _render_probability_bars(probs, emotions_list):
    """Menggambar grafik horizontal modern untuk distribusi probabilitas emosi."""
    bar_colors = {
        "angry": "#e74c3c", "disgust": "#8e44ad", "fear": "#e67e22",
        "happy": "#f1c40f", "sad": "#3498db", "surprise": "#1abc9c",
        "neutral": "#95a5a6",
    }
    sorted_pairs = sorted(zip(emotions_list, probs), key=lambda p: p[1], reverse=True)
    html_parts = []
    for emo, prob in sorted_pairs:
        pct = prob * 100
        color = bar_colors.get(emo, "#1DB954")
        emoji = EMOTION_EMOJI.get(emo, "")
        html_parts.append(
            f'<div class="emo-bar-wrap">'
            f'<span class="emo-bar-label">{emoji} {emo.capitalize()}</span>'
            f'<span class="emo-bar-track">'
            f'<span class="emo-bar-fill" style="width:{pct:.1f}%;background:{color};"></span>'
            f"</span>"
            f'<span class="emo-bar-pct">{pct:.1f}%</span>'
            f"</div>"
        )
    st.markdown("".join(html_parts), unsafe_allow_html=True)


def _render_profile_mapping(profile, emotion_label):
    """Menampilkan detail target pemetaan audio features dari emosi yang terdeteksi."""
    feature_labels = {
        "valence": "Valensi (Valence)", "energy": "Energi (Energy)",
        "danceability": "Dansabilitas (Danceability)", "tempo": "Tempo (BPM)",
        "acousticness": "Akustik (Acousticness)", "instrumentalness": "Instrumental",
        "liveness": "Keaktifan (Liveness)", "speechiness": "Vokal (Speechiness)",
    }
    rows = "".join(
        f'<div class="profile-row">'
        f"<span>{feature_labels.get(k, k)}</span>"
        f'<span class="profile-val">{v}</span>'
        f"</div>"
        for k, v in profile.items()
    )
    st.markdown(
        f'<div class="profile-map">'
        f"<h4>Karakteristik Target Musik untuk {EMOTION_EMOJI.get(emotion_label, '')} "
        f"{emotion_label.capitalize()}</h4>"
        f"{rows}</div>",
        unsafe_allow_html=True,
    )

# FUNGSI UTAMA (MAIN APP FLOW)

def main():
    emotion_model, emotion_metadata = load_emotion_model()
    knn_model, scaler, rec_metadata = load_recommender()

    if knn_model is None or scaler is None or rec_metadata is None:
        st.error(
            "Gagal memuat model rekomendasi. Pastikan file berikut ada di "
            "`model/model rekomendasi musik/`: `knn_model.pkl`, `scaler.pkl`, `recommender_metadata.json`, `track_db.csv`"
        )
        return

    track_df = load_track_database()
    if track_df.empty:
        st.error("Database musik kosong atau tidak dapat dimuat.")
        return

    emotion_profiles = rec_metadata.get("emotion_profiles", {})
    tempo_idx = FEATURE_COLS.index("tempo")
    tempo_min_raw = float(scaler.data_min_[tempo_idx])
    tempo_max_raw = float(scaler.data_max_[tempo_idx])

    # Inisialisasi session state untuk offset navigasi halaman rekomendasi
    if "recommendation_offset" not in st.session_state:
        st.session_state.recommendation_offset = 0

    # Callback untuk reset offset rekomendasi ke halaman 1 ketika input diubah
    def reset_offset():
        st.session_state.recommendation_offset = 0

    # Header Aplikasi
    st.markdown(
        """
    <div class="main-header">
        <h1>SISTEM REKOMENDASI MUSIK BERBASIS DETEKSI EMOSI WAJAH DAN FITUR AUDIO</h1>
        <p>Sistem ini merekomendasikan musik secara personal berdasarkan <b>ekspresi emosi wajah</b> Anda dan preferensi karakteristik audio. Gunakan kamera untuk deteksi ekspresi wajah otomatis, atau pilih suasana hati secara manual untuk mulai menjelajahi playlist rekomendasi musik!</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

     # CONTAINER KAMERA & DETEKSI EMOSI
   
    camera_detected_emotion = None
    camera_confidence = 0.0
    camera_all_probs = None

    if "detected_emotion" not in st.session_state:
        st.session_state.detected_emotion = None
        st.session_state.detected_confidence = 0.0
        st.session_state.detected_probs = None
        st.session_state.detected_annotated = None

    cam_col_left, cam_col_right = st.columns([1.1, 0.9])

    with cam_col_left:
        st.markdown(
            '<div class="camera-section">'
            "<h3>📸 Deteksi Emosi via Kamera</h3>"
            "<p style='color:rgba(255,255,255,0.6);font-size:0.9rem;margin-bottom:12px;'>"
            "Ambil foto wajah menggunakan kamera untuk deteksi emosi otomatis via model CNN+CBAM.</p>"
            "</div>",
            unsafe_allow_html=True,
        )

        camera_available = TF_AVAILABLE and emotion_model is not None

        if camera_available:
            camera_photo = st.camera_input(
                "Posisikan wajah Anda di tengah kamera",
                key="cam_input",
                on_change=reset_offset,
            )

            if camera_photo is not None:
                img = Image.open(camera_photo)
                img_arr = np.array(img)
                emo_list = emotion_metadata.get("emotions", EMOTIONS)

                with st.spinner("Menganalisis ekspresi wajah..."):
                    det_label, det_conf, det_probs, det_annotated = (
                        detect_emotion_from_image(img_arr, emotion_model, emo_list)
                    )

                if det_label:
                    st.session_state.detected_emotion = det_label
                    st.session_state.detected_confidence = det_conf
                    st.session_state.detected_probs = det_probs
                    st.session_state.detected_annotated = det_annotated
                else:
                    st.warning(
                        "Wajah tidak terdeteksi. Silakan coba lagi dengan memastikan wajah terlihat jelas dan mendapatkan pencahayaan yang cukup."
                    )
        else:
            st.info(
                "Deteksi emosi berbasis kamera memerlukan pustaka **TensorFlow**. Hubungkan kamera atau gunakan panel manual di bawah untuk memilih emosi."
            )

    with cam_col_right:
        if st.session_state.detected_emotion is not None:
            det_emo = st.session_state.detected_emotion
            det_conf = st.session_state.detected_confidence
            det_probs = st.session_state.detected_probs
            det_ann = st.session_state.detected_annotated

            st.markdown(
                f'<div class="detection-result">'
                f'<span class="emoji">{EMOTION_EMOJI.get(det_emo, "🎭")}</span>'
                f'<span class="label">{det_emo.capitalize()}</span>'
                f'<span class="conf">Tingkat Keyakinan: {det_conf:.1%}</span>'
                f"</div>",
                unsafe_allow_html=True,
            )

            if det_ann is not None:
                st.image(det_ann, caption="Visual Hasil Deteksi Wajah", use_container_width=True)

            if det_probs is not None:
                emo_list = emotion_metadata.get("emotions", EMOTIONS)
                st.markdown("**Distribusi Probabilitas Emosi:**")
                _render_probability_bars(det_probs, emo_list)

            if det_emo in emotion_profiles:
                _render_profile_mapping(emotion_profiles[det_emo], det_emo)

            camera_detected_emotion = det_emo
            camera_confidence = det_conf
        else:
            st.markdown(
                '<div style="text-align:center;padding:60px 20px;'
                "color:rgba(255,255,255,0.35);font-size:1.1rem;\">"
                "📸 Ambil foto wajah Anda untuk melihat hasil deteksi emosi di sini."
                "</div>",
                unsafe_allow_html=True,
            )

    st.markdown(
        '<hr style="border-color:rgba(255,255,255,0.06);margin:8px 0 24px;">',
        unsafe_allow_html=True,
    )

       # PANEL KONTROL KUSTOMISASI AUDIO
   
    col_features, col_genre, col_emotion = st.columns([3, 1.4, 1.4])

    with col_emotion:
        st.markdown(
            '<p class="section-header">Pilih Suasana Hati (Emosi):</p>',
            unsafe_allow_html=True,
        )

        emotion_options = [
            "Angry", "Disgust", "Fear", "Happy",
            "Sad", "Surprise", "Neutral",
        ]

        if camera_detected_emotion:
            default_idx = next(
                (i for i, e in enumerate(emotion_options) if e.lower() == camera_detected_emotion),
                6,
            )
        else:
            default_idx = 6

        selected_option = st.selectbox(
            "Emosi",
            emotion_options,
            index=default_idx,
            label_visibility="collapsed",
            key="emotion_selector",
            on_change=reset_offset,
        )
        current_emotion = selected_option.lower()

        if camera_detected_emotion:
            current_emotion = camera_detected_emotion

        emoji = EMOTION_EMOJI.get(current_emotion, "🎭")
        source = "Kamera" if camera_detected_emotion == current_emotion and camera_detected_emotion else "Manual"

        st.markdown(
            f'<div class="emotion-badge">'
            f"<span>{emoji} {current_emotion.capitalize()}</span><br>"
            f'<span style="font-size:0.78rem;color:rgba(255,255,255,0.45);">'
            f"Metode: {source}"
            f"{' | Akurasi: ' + f'{camera_confidence:.0%}' if source == 'Kamera' else ''}"
            f"</span></div>",
            unsafe_allow_html=True,
        )

        if camera_detected_emotion:
            st.markdown(
                '<div class="sync-arrow">&#8595; disinkronkan dengan rekomendasi</div>',
                unsafe_allow_html=True,
            )

    with col_genre:
        st.markdown(
            '<p class="section-header">Pilih Genre Musik:</p>',
            unsafe_allow_html=True,
        )
        genre_selected = st.radio(
            "Genre",
            TARGET_GENRES,
            index=7,
            label_visibility="collapsed",
            key="genre_selector",
            on_change=reset_offset,
        )

    with col_features:
        st.markdown(
            '<p class="section-header">Sesuaikan Karakteristik Audio:</p>',
            unsafe_allow_html=True,
        )

        profile = emotion_profiles.get(current_emotion, emotion_profiles.get("neutral", {}))

        has_year = "release_year" in track_df.columns and track_df["release_year"].notna().any()
        if has_year:
            yr_min = int(track_df["release_year"].min())
            yr_max = int(track_df["release_year"].max())
        else:
            yr_min, yr_max = 1990, 2023

        year_range = st.slider(
            "Rentang Tahun Rilis",
            yr_min,
            yr_max,
            (yr_min, yr_max),
            key=f"yr_{current_emotion}",
            on_change=reset_offset,
        )
        if not has_year:
            year_range = None

        sl1, sl2 = st.columns(2)

        with sl1:
            acousticness = st.slider(
                "Akustik (Acousticness)", 0.0, 1.0,
                float(profile.get("acousticness", 0.5)),
                step=0.01, key=f"ac_{current_emotion}",
                on_change=reset_offset,
            )
            danceability = st.slider(
                "Dansabilitas (Danceability)", 0.0, 1.0,
                float(profile.get("danceability", 0.5)),
                step=0.01, key=f"da_{current_emotion}",
                on_change=reset_offset,
            )
            energy = st.slider(
                "Energi (Energy)", 0.0, 1.0,
                float(profile.get("energy", 0.5)),
                step=0.01, key=f"en_{current_emotion}",
                on_change=reset_offset,
            )
            instrumentalness = st.slider(
                "Instrumental (Instrumentalness)", 0.0, 1.0,
                float(profile.get("instrumentalness", 0.2)),
                step=0.01, key=f"in_{current_emotion}",
                on_change=reset_offset,
            )

        with sl2:
            liveness = st.slider(
                "Keaktifan Live (Liveness)", 0.0, 1.0,
                float(profile.get("liveness", 0.25)),
                step=0.01, key=f"li_{current_emotion}",
                on_change=reset_offset,
            )
            speechiness = st.slider(
                "Vokal (Speechiness)", 0.0, 1.0,
                float(profile.get("speechiness", 0.1)),
                step=0.01, key=f"sp_{current_emotion}",
                on_change=reset_offset,
            )
            valence = st.slider(
                "Valensi Keceriaan (Valence)", 0.0, 1.0,
                float(profile.get("valence", 0.5)),
                step=0.01, key=f"va_{current_emotion}",
                on_change=reset_offset,
            )

            tempo_scaled_default = float(profile.get("tempo", 0.5))
            tempo_raw_default = tempo_scaled_default * (tempo_max_raw - tempo_min_raw) + tempo_min_raw
            tempo_raw_default = max(tempo_min_raw, min(tempo_max_raw, tempo_raw_default))
            tempo = st.slider(
                "Tempo (BPM)",
                round(tempo_min_raw, 2),
                round(tempo_max_raw, 2),
                round(tempo_raw_default, 2),
                step=0.5,
                key=f"te_{current_emotion}",
                on_change=reset_offset,
            )

    st.markdown(
        '<hr style="border-color: rgba(255,255,255,0.06);">',
        unsafe_allow_html=True,
    )

    # HASIL REKOMENDASI MUSIK
   
    header_source = " (kamera)" if camera_detected_emotion else ""
    header_text = f"Rekomendasi Musik untuk Suasana Hati {EMOTION_EMOJI.get(current_emotion, '')} {current_emotion.capitalize()}{header_source}"
    st.markdown(
        f'<div class="results-header"><h2>{header_text}</h2></div>',
        unsafe_allow_html=True,
    )

    feature_overrides = {
        "valence": valence,
        "energy": energy,
        "danceability": danceability,
        "tempo": tempo,
        "acousticness": acousticness,
        "instrumentalness": instrumentalness,
        "liveness": liveness,
        "speechiness": speechiness,
    }

    # Kueri kumpulan lagu yang lebih besar (200 lagu) untuk mendukung fitur "Rekomendasi Lainnya"
    with st.spinner("Mencari lagu terbaik untuk suasana hati Anda..."):
        results = recommend_songs(
            emotion=current_emotion,
            df=track_df,
            scaler=scaler,
            metadata=rec_metadata,
            genre=genre_selected,
            n_recs=len(track_df),
            year_range=year_range,
            **feature_overrides,
        )

    if results.empty:
        st.warning(
            "Tidak ada lagu yang cocok dengan kriteria filter Anda. Coba sesuaikan pilihan genre musik atau karakteristik audio."
        )
        return

    total_recs = len(results)
    offset = st.session_state.recommendation_offset

    # Pastikan offset aman dan tidak melebihi jumlah hasil kueri
    if offset >= total_recs:
        offset = 0
        st.session_state.recommendation_offset = 0

    # Ambil subset 10 lagu yang aktif sesuai dengan offset
    display_results = results.iloc[offset : offset + N_DISPLAY_CARDS]

    st.caption(
        f"Menampilkan rekomendasi ke {offset + 1} - {min(offset + len(display_results), total_recs)} dari {total_recs} musik teratas untuk Suasana Hati **{current_emotion.capitalize()}** | Genre **{genre_selected}**"
    )

    # Merender kartu lagu dalam tata letak 2 kolom
    card_cols = st.columns(2)

    for idx, (_, track) in enumerate(display_results.iterrows()):
        actual_idx = offset + idx
        with card_cols[idx % 2]:
            # Dibungkus dengan st.container ber-key unik untuk stabilitas DOM virtual React
            with st.container(key=f"card_container_{track['id']}_{actual_idx}"):
                render_song_card(
                    track_id=str(track["id"]),
                    name=str(track["name"]),
                    artist=str(track.get("artists_name", "Unknown Artist")),
                    similarity=float(track["similarity"]),
                    card_idx=actual_idx,
                )

                with st.expander("Lihat Detail Karakteristik Audio"):
                    d1, d2 = st.columns(2)
                    with d1:
                        st.metric("Valensi (Valence)", f"{track['valence']:.3f}")
                        st.metric("Energi (Energy)", f"{track['energy']:.3f}")
                        st.metric("Dansabilitas (Danceability)", f"{track['danceability']:.3f}")
                        st.metric("Instrumental", f"{track['instrumentalness']:.5f}")
                    with d2:
                        st.metric("Speechiness (Vokal)", f"{track['speechiness']:.3f}")
                        st.metric("Keaktifan Live (Liveness)", f"{track['liveness']:.3f}")
                        st.metric("Tempo", f"{track['tempo']:.1f} BPM")
                        st.metric("Akustik", f"{track['acousticness']:.3f}")

                    st.markdown(f"**Genre:** {track.get('genres', 'N/A')}")
                    st.markdown(f"**Popularitas:** {track.get('popularity', 'N/A')}")
                    st.markdown(
                        f"[Buka di Spotify](https://open.spotify.com/track/{track['id']})"
                    )

    # Navigasi ke halaman rekomendasi lagu alternatif lainnya (jika total rekomendasi > 10)
    if total_recs > N_DISPLAY_CARDS:
        st.markdown("<br>", unsafe_allow_html=True)
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("⏮️ Kembali ke Awal (Terbaik)", use_container_width=True, key="reset_recs_button"):
                st.session_state.recommendation_offset = 0
                st.rerun()
        with btn_col2:
            if st.button("🔄 Tampilkan Rekomendasi Musik Lainnya", use_container_width=True, key="next_recs_button"):
                st.session_state.recommendation_offset = (offset + N_DISPLAY_CARDS) % total_recs
                st.rerun()


if __name__ == "__main__":
    main()
