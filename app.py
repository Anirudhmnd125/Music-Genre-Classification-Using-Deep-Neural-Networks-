"""Streamlit Web Application for Music Genre Classification.

Provides an interactive GUI to:
- Test trained models (MLP, CNN, CRNN)
- Upload or select audio samples
- Play audio and view waveform / spectrogram
- Inspect model confidence scores across genres
- View performance metrics, confusion matrices and model comparison
"""
import os
import glob
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import librosa
import librosa.display

from src.config import AudioConfig
from src.evaluate import load_checkpoint, predict_proba, compare_models
from src.features import extract_file_features, load_audio


st.set_page_config(
    page_title="Music Genre Classifier",
    page_icon="🎵",
    layout="wide"
)

st.title("🎵 Music Genre Classification with Deep Neural Networks")
st.markdown(
    "Classify music tracks into genres using deep learning architectures: "
    "**MLP** (Multi-Layer Perceptron), **CNN** (Convolutional Neural Network), and **CRNN** (Convolutional Recurrent Neural Network)."
)

# Sidebar: Model and settings
st.sidebar.header("⚙️ Configuration")

checkpoint_dir = st.sidebar.text_input("Outputs Directory", value="outputs/demo")
available_ckpts = glob.glob(os.path.join(checkpoint_dir, "*_best.pt"))

if not available_ckpts:
    st.sidebar.warning(f"No checkpoint files found in `{checkpoint_dir}`. Please run demo training first.")
    selected_ckpt = None
else:
    ckpt_names = {os.path.basename(p): p for p in sorted(available_ckpts)}
    selected_name = st.sidebar.selectbox("Select Model Checkpoint", list(ckpt_names.keys()), index=0)
    selected_ckpt = ckpt_names[selected_name]

# Main tabs
tab_predict, tab_compare, tab_metrics = st.tabs(["🎧 Audio Inference", "📊 Model Comparison", "📈 Training & Metrics"])

with tab_predict:
    st.subheader("1. Select or Upload Audio")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        input_mode = st.radio("Audio Source", ["Preset Demo Track", "Upload Custom Audio File"], horizontal=True)
    
    audio_file_path = None
    temp_path = None

    if input_mode == "Preset Demo Track":
        demo_tracks = glob.glob("data/demo/*/*.wav")
        if demo_tracks:
            selected_track = st.selectbox("Choose a demo audio sample", sorted(demo_tracks))
            audio_file_path = selected_track
        else:
            st.info("No demo files found in data/demo. Run `python main.py demo-data` to generate them.")
    else:
        uploaded = st.file_uploader("Upload an audio file (.wav, .mp3, .ogg, .flac)", type=["wav", "mp3", "ogg", "flac"])
        if uploaded is not None:
            temp_path = os.path.join("data", "uploaded_" + uploaded.name)
            os.makedirs("data", exist_ok=True)
            with open(temp_path, "wb") as f:
                f.write(uploaded.read())
            audio_file_path = temp_path

    if audio_file_path and os.path.isfile(audio_file_path):
        st.write("---")
        st.subheader("2. Audio Playback & Analysis")
        
        col_player, col_meta = st.columns([2, 1])
        with col_player:
            st.audio(audio_file_path)
        with col_meta:
            st.caption(f"**File:** `{os.path.basename(audio_file_path)}`")
            file_size_kb = os.path.getsize(audio_file_path) / 1024
            st.caption(f"**Size:** {file_size_kb:.1f} KB")

        # Load audio for visualization
        y_signal, sr = librosa.load(audio_file_path, sr=22050, mono=True)
        duration_s = len(y_signal) / sr
        st.caption(f"**Duration:** {duration_s:.1f}s | **Sampling Rate:** {sr} Hz")

        # Waveform and Spectrogram Visualizations
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4.5))
        librosa.display.waveshow(y_signal, sr=sr, ax=ax1, color="#1f77b4")
        ax1.set(title="Waveform", xlabel="Time (s)", ylabel="Amplitude")
        ax1.grid(alpha=0.3)

        mel = librosa.feature.melspectrogram(y=y_signal, sr=sr, n_mels=64)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        img = librosa.display.specshow(mel_db, sr=sr, x_axis="time", y_axis="mel", ax=ax2, cmap="magma")
        ax2.set(title="Log-Mel Spectrogram")
        fig.colorbar(img, ax=ax2, format="%+2.0f dB")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Run Prediction
        st.write("---")
        st.subheader("3. Model Genre Prediction")

        if selected_ckpt is None:
            st.error("No model checkpoint selected.")
        else:
            if st.button("🚀 Predict Genre", type="primary"):
                with st.spinner("Extracting features and predicting..."):
                    model, ckpt, device = load_checkpoint(selected_ckpt)
                    cfg = AudioConfig(**ckpt["audio_cfg"])
                    feats = extract_file_features(audio_file_path, cfg)
                    probs_all = predict_proba(model, feats, ckpt["mean"], ckpt["std"], device)
                    track_probs = probs_all.mean(0)
                    genres = ckpt["genres"]
                    top_idx = int(track_probs.argmax())
                    top_genre = genres[top_idx]
                    top_conf = float(track_probs[top_idx])

                st.success(f"### Predicted Genre: **{top_genre.upper()}** ({top_conf:.1%})")
                
                # Probability bars
                st.write("**Full Genre Probability Distribution:**")
                sorted_indices = np.argsort(track_probs)[::-1]
                
                prog_cols = st.columns(2)
                for rank, idx in enumerate(sorted_indices):
                    g = genres[idx]
                    p = float(track_probs[idx])
                    target_col = prog_cols[rank % 2]
                    with target_col:
                        target_col.write(f"**{g.capitalize()}**: {p:.1%}")
                        target_col.progress(min(max(p, 0.0), 1.0))

                # Segment-level progression
                if len(feats) > 1:
                    with st.expander(f"Detailed breakdown across {len(feats)} track segments (3s each)"):
                        seg_genres = [genres[int(p.argmax())] for p in probs_all]
                        st.write("Segment-by-segment predictions:")
                        st.write(", ".join([f"Segment {i+1}: **{sg}**" for i, sg in enumerate(seg_genres)]))

    # Clean up temp file
    if temp_path and os.path.exists(temp_path):
        try:
            os.remove(temp_path)
        except Exception:
            pass

with tab_compare:
    st.subheader("Model Benchmark & Architecture Comparison")
    st.markdown("Compares test performance, parameter count, and inference latency across trained architectures.")
    
    if available_ckpts:
        if st.button("🔄 Run Side-by-Side Comparison"):
            with st.spinner("Evaluating checkpoints..."):
                results = compare_models(available_ckpts)
                if results:
                    st.dataframe(results, use_container_width=True)
    else:
        st.info("Train at least one model checkpoint to see the comparison table.")

with tab_metrics:
    st.subheader("Model Artifacts & Performance Reports")
    if selected_ckpt:
        model_name = os.path.basename(selected_ckpt).replace("_best.pt", "")
        out_dir = os.path.dirname(selected_ckpt)
        
        report_file = os.path.join(out_dir, f"{model_name}_report.txt")
        cm_file = os.path.join(out_dir, f"{model_name}_confusion_matrix.png")
        curves_file = os.path.join(out_dir, f"{model_name}_training_curves.png")

        col_left, col_right = st.columns(2)
        with col_left:
            if os.path.isfile(curves_file):
                st.image(curves_file, caption=f"{model_name.upper()} Training Curves")
            if os.path.isfile(cm_file):
                st.image(cm_file, caption=f"{model_name.upper()} Confusion Matrix")

        with col_right:
            if os.path.isfile(report_file):
                st.write(f"**Classification Report ({model_name.upper()}):**")
                with open(report_file, "r") as f:
                    st.code(f.read(), language="text")
            else:
                st.info(f"No report file found for {model_name}. Run `python main.py evaluate --checkpoint {selected_ckpt}`.")
