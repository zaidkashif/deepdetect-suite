import os
import numpy as np
import streamlit as st
import librosa
import joblib
import soundfile as sf
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
st.set_page_config(page_title="AI Predictive Tools", layout="wide")

# ===================
# Define Directories
# ===================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PART1_MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'Part1', 'part1_artifacts'))
PART2_MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'Part2', 'part2_artifacts'))



# ===================
# Safe Model Loading
# ===================
def safe_load_model(path):
    try:
        return joblib.load(path)
    except FileNotFoundError:
        st.warning(f"⚠️ Model file not found: {path}")
        return None

def safe_load_config(path):
    try:
        with open(path, "rb") as f:
            return joblib.load(f)
    except FileNotFoundError:
        st.warning(f"⚠️ Config file not found: {path}")
        return None

# ===================
# Load Part 1 Models (Audio Deepfake)
# ===================
audio_models = {
    "Logistic Regression": safe_load_model(os.path.join(PART1_MODEL_DIR, "logistic_regression_p1.joblib")),
    "SVM": safe_load_model(os.path.join(PART1_MODEL_DIR, "svm_p1.joblib")),
    "Perceptron": safe_load_model(os.path.join(PART1_MODEL_DIR, "perceptron_p1.joblib")),
}
scaler_p1 = safe_load_model(os.path.join(PART1_MODEL_DIR, "scaler_p1.joblib"))
config_p1 = safe_load_config(os.path.join(PART1_MODEL_DIR, "part1_config.pkl"))

# ===================
# Load Part 2 Models (Software Defects)
# ===================
text_models = {
    "Logistic Regression": safe_load_model(os.path.join(PART2_MODEL_DIR, "logistic_regression_p2.joblib")),
    "SVM": safe_load_model(os.path.join(PART2_MODEL_DIR, "svm_p2.joblib")),
    "Perceptron": safe_load_model(os.path.join(PART2_MODEL_DIR, "perceptron_p2.joblib")),
}
vectorizer = safe_load_model(os.path.join(PART2_MODEL_DIR, "tfidf_vectorizer.joblib"))
config_p2 = safe_load_config(os.path.join(PART2_MODEL_DIR, "part2_config.pkl"))

# ===================
# Streamlit UI Layout
# ===================

# 🔍 Warn about any missing models/configs
missing = [k for k, v in {**audio_models, **text_models, "Scaler": scaler_p1, "Vectorizer": vectorizer}.items() if v is None]
if missing:
    st.warning(f"⚠️ Missing components: {', '.join(missing)}")


st.markdown(
    """
    <style>
        .main { background-color: #f7f9fa; }
        h1, h2, h3 { color: #1a4d6e; }
        .stButton>button { background-color: #1a4d6e; color: white; border-radius: 6px; }
        .stSelectbox>div>div>div>div { color: #1a4d6e; font-weight: 600; }
    </style>
    """, unsafe_allow_html=True
)

st.title("🔍 AI-based Detection Suite")

tabs = st.tabs(["🔊 Deepfake Audio Detection", "🐞 Software Defect Prediction"])

# =============================
# Tab 1: Audio Deepfake Detection
# =============================
with tabs[0]:
    st.header("🔊 Urdu Deepfake Audio Detection")

    uploaded_file = st.file_uploader("Upload a short Urdu audio file (WAV/FLAC)", type=["wav", "flac"])
    selected_audio_model = st.selectbox("Select Model", list(audio_models.keys()))
    predict_audio = st.button("🎧 Analyze Audio")

    if predict_audio and uploaded_file:
        model = audio_models[selected_audio_model]
        if model is None or scaler_p1 is None or config_p1 is None:
            st.error("❌ Required model or config is missing. Please check file availability.")
        else:
            audio, sr = sf.read(uploaded_file)
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            max_len = int(config_p1['MAX_AUDIO_LEN_SECONDS'] * sr)
            audio = librosa.util.fix_length(audio, max_len)
            mfcc = librosa.feature.mfcc(y=audio.astype(np.float32), sr=sr, n_mfcc=config_p1['N_MFCC'])
            features = np.mean(mfcc.T, axis=0).reshape(1, -1)
            features_scaled = scaler_p1.transform(features)
            pred = model.predict(features_scaled)[0]
            result_text = "✅ Bonafide (Real)" if pred == 0 else "⚠️ Deepfake"
            st.success(f"**Prediction:** {result_text}")

# =============================
# Tab 2: Text Defect Prediction
# =============================
with tabs[1]:
    st.header("🐞 Multi-label Software Defect Prediction")
    bug_text = st.text_area("Enter the bug report below:", height=150, placeholder="e.g. The app crashes after update...")
    selected_text_model = st.selectbox("Select Model", list(text_models.keys()), key="text_model")
    predict_text = st.button("🧠 Predict Defects")

    if predict_text and bug_text:
        model = text_models[selected_text_model]
        if model is None or vectorizer is None or config_p2 is None:
            st.error("❌ Required model or config is missing. Please check file availability.")
        else:
            tfidf_input = vectorizer.transform([bug_text])
            prediction = model.predict(tfidf_input)[0]
            labels = config_p2['labels']
            predicted_labels = [label for label, value in zip(labels, prediction) if value == 1]
            st.subheader("🔎 Predicted Labels:")
            if predicted_labels:
                for label in predicted_labels:
                    st.success(f"• {label}")
            else:
                st.warning("No defect labels predicted.")
