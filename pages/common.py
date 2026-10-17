"""Shared Streamlit workflows for single-character input and results."""

from __future__ import annotations

import io
import json
from datetime import datetime, timezone

import numpy as np
import plotly.express as px
import streamlit as st
from PIL import Image, ImageOps

from src.inference import predict_top_k
from src.preprocessing import preprocess_image

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_SUFFIXES = {"png", "jpg", "jpeg"}


def validate_upload(uploaded_file) -> Image.Image:
    if uploaded_file is None:
        raise ValueError("Choose an image file to continue.")
    suffix = uploaded_file.name.rsplit(".", 1)[-1].lower() if "." in uploaded_file.name else ""
    if suffix not in ALLOWED_SUFFIXES:
        raise ValueError("Use a PNG, JPG, or JPEG image.")
    if uploaded_file.size > MAX_UPLOAD_BYTES:
        raise ValueError("Image files must be 10 MB or smaller.")
    try:
        uploaded_file.seek(0)
        image = Image.open(uploaded_file)
        image.verify()
        uploaded_file.seek(0)
        return ImageOps.exif_transpose(Image.open(uploaded_file)).convert("RGB")
    except Exception as error:
        raise ValueError("This image could not be read. Try a valid PNG or JPEG file.") from error


def reject_multiple_characters(image: Image.Image) -> None:
    grayscale = np.asarray(ImageOps.grayscale(image), dtype=np.uint8)
    if grayscale.mean() < 127:
        foreground = grayscale > max(20, int(grayscale.max() * 0.15))
    else:
        foreground = grayscale < min(235, int(grayscale.min() + (grayscale.max() - grayscale.min()) * 0.85))
    ys, xs = np.where(foreground)
    if len(xs) == 0:
        raise ValueError("No handwritten character was detected in this image.")
    if (xs.max() - xs.min() + 1) / max(1, ys.max() - ys.min() + 1) > 2.3:
        raise ValueError("This image appears to contain multiple characters. Upload one letter per image.")


def get_top_predictions(model, image: Image.Image | np.ndarray):
    checked_image = image if isinstance(image, Image.Image) else Image.fromarray(image)
    reject_multiple_characters(checked_image)
    prepared = preprocess_image(image)
    return prepared, predict_top_k(model, prepared, k=3)


def show_predictions(model, image: Image.Image | np.ndarray, source: str, thumbnail=None) -> dict:
    prepared, predictions = get_top_predictions(model, image)
    letter, confidence = predictions[0]
    st.session_state.setdefault("prediction_history", []).append(
        {
            "number": len(st.session_state["prediction_history"]) + 1,
            "timestamp": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "source": source,
            "character": letter,
            "confidence": confidence,
            "thumbnail": thumbnail,
        }
    )
    left, right = st.columns([1, 1.2])
    with left:
        st.markdown("#### Processed input")
        st.image(prepared.squeeze(), width=170, clamp=True)
        st.caption("28 × 28 · grayscale · normalized")
    with right:
        st.markdown("#### Model output")
        st.markdown(f"<div class='prediction-letter'>{letter}</div>", unsafe_allow_html=True)
        st.metric("Top match · softmax score", f"{confidence:.1%}")
        chart = px.bar(
            x=[score for _, score in predictions],
            y=[label for label, _ in predictions],
            orientation="h",
            labels={"x": "Score", "y": "Letter"},
            range_x=[0, 1],
            color=[score for _, score in predictions],
            color_continuous_scale=[[0, "#47d6c5"], [1, "#4b8df8"]],
        )
        chart.update_layout(height=190, margin=dict(l=8, r=8, t=8, b=8), showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(chart, use_container_width=True, config={"displayModeBar": False})
    return {
        "timestamp": st.session_state["prediction_history"][-1]["timestamp"],
        "source": source,
        "character": letter,
        "confidence": confidence,
        "top_three": [{"character": label, "score": score} for label, score in predictions],
    }


def result_downloads(result: dict, key: str) -> None:
    left, right = st.columns(2)
    payload = json.dumps(result, indent=2)
    with left:
        st.download_button("Download JSON", payload, f"{key}.json", "application/json", use_container_width=True)
    with right:
        import pandas as pd

        st.download_button(
            "Download CSV",
            pd.DataFrame(result["top_three"]).to_csv(index=False),
            f"{key}.csv",
            "text/csv",
            use_container_width=True,
        )