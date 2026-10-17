"""Current Streamlit session prediction log."""

import pandas as pd
import streamlit as st

from src.utils import rows_to_csv


def render(get_model) -> None:
    st.markdown("## Prediction history")
    st.caption("This history is kept only in the current Streamlit session and may disappear when it ends.")
    records = st.session_state.get("prediction_history", [])
    if not records:
        st.info("Predictions from drawing and uploads will appear here.")
        return
    controls = st.columns([2, 1, 1])
    query = controls[0].text_input("Search letter or time", placeholder="Search this session")
    sources = ["All sources"] + sorted({record["source"] for record in records})
    source = controls[1].selectbox("Source", sources)
    controls[2].metric("Total", len(records))
    filtered = [
        record
        for record in records
        if (source == "All sources" or record["source"] == source)
        and (query.lower() in record["character"].lower() or query.lower() in record["timestamp"].lower())
    ]
    display = [
        {key: item[key] for key in ("number", "timestamp", "source", "character", "confidence")}
        for item in filtered
    ]
    columns = st.columns(3)
    columns[0].metric("Matching predictions", len(filtered))
    columns[1].metric("Mean top score", f"{sum(row['confidence'] for row in filtered) / len(filtered):.1%}" if filtered else "—")
    columns[2].metric("Distinct letters", len({row["character"] for row in filtered}))
    st.dataframe(pd.DataFrame(display), use_container_width=True, hide_index=True)
    st.download_button("Export history as CSV", rows_to_csv(display), "glyphsense_history.csv", "text/csv")
    for record in reversed(filtered):
        if record.get("thumbnail") is not None:
            st.image(record["thumbnail"], caption=f"{record['character']} · {record['source']}", width=80)
    if st.button("Clear history", type="secondary"):
        st.session_state["prediction_history"] = []
        st.rerun()