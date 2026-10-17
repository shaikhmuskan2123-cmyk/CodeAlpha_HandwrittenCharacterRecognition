"""GlyphSense AI Streamlit application."""

from pathlib import Path

import streamlit as st

from config import ASSET_DIR, MODEL_PATH
from pages import about, analytics, batch_predict, draw_predict, history, overview, upload_predict
from src.inference import load_model

st.set_page_config(
    page_title="GlyphSense AI",
    page_icon="✳",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=False)
def get_model():
    return load_model()


def main() -> None:
    css_path = ASSET_DIR / "style.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("<div class='brand-mark'>G</div>", unsafe_allow_html=True)
        st.markdown("### GlyphSense AI")
        st.caption("HANDWRITTEN CHARACTER INTELLIGENCE")
        selected = st.radio(
            "Workspace",
            [
                "Overview",
                "Draw & Predict",
                "Upload & Predict",
                "Batch Recognition",
                "Model Analytics",
                "Prediction History",
                "About the Project",
            ],
            label_visibility="collapsed",
        )
        st.divider()
        ready = Path(MODEL_PATH).is_file()
        status = "Model ready" if ready else "Model not trained"
        st.markdown(
            f"<div class='sidebar-status'><span class='status-dot {'ready' if ready else ''}'></span>{status}</div>",
            unsafe_allow_html=True,
        )
        st.caption("EMNIST Letters · 26 classes")

    pages = {
        "Overview": overview,
        "Draw & Predict": draw_predict,
        "Upload & Predict": upload_predict,
        "Batch Recognition": batch_predict,
        "Model Analytics": analytics,
        "Prediction History": history,
        "About the Project": about,
    }
    pages[selected].render(get_model)


if __name__ == "__main__":
    main()