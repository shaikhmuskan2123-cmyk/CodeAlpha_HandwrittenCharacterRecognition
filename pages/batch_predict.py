"""Batch recognition with per-image error isolation and CSV export."""

import pandas as pd
import streamlit as st

from pages.common import get_top_predictions, validate_upload
from src.inference import load_model
from src.preprocessing import preprocess_image
from src.utils import utc_timestamp


def render(get_model) -> None:
    st.markdown("## Batch recognition")
    st.caption("Upload up to 50 images. Each file must contain one handwritten English letter.")
    files = st.file_uploader(
        "Choose images",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key="batch_upload",
    )
    if len(files) > 50:
        st.error("Select no more than 50 images per batch.")
        return
    if not files or not st.button("Process batch", type="primary"):
        return
    try:
        model = get_model()
    except FileNotFoundError as error:
        st.error(str(error))
        return

    results = []
    for file in files:
        record = {"filename": file.name, "predicted_letter": "", "confidence": "", "status": ""}
        try:
            image = validate_upload(file)
            _, predictions = get_top_predictions(model, image)
            record["predicted_letter"] = predictions[0][0]
            record["confidence"] = f"{predictions[0][1]:.6f}"
            record["status"] = "ok"
            st.session_state.setdefault("prediction_history", []).append(
                {
                    "number": len(st.session_state["prediction_history"]) + 1,
                    "timestamp": utc_timestamp(),
                    "source": "Batch upload",
                    "character": predictions[0][0],
                    "confidence": predictions[0][1],
                    "thumbnail": None,
                }
            )
        except Exception as error:
            record["status"] = f"error: {error}"
        results.append(record)

    st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
    st.download_button(
        "Download batch CSV",
        pd.DataFrame(results).to_csv(index=False),
        "glyphsense_batch_results.csv",
        "text/csv",
    )