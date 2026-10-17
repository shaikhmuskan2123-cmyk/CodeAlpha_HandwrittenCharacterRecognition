"""Project status and quick-start dashboard."""

import plotly.express as px
import streamlit as st

from config import HISTORY_PATH, METRICS_PATH, MODEL_PATH
from src.dashboard_data import class_test_profile
from src.utils import load_json


def render(get_model) -> None:
    metrics = load_json(METRICS_PATH)
    training = load_json(HISTORY_PATH)
    ready = MODEL_PATH.is_file()
    records = st.session_state.get("prediction_history", [])
    status = "Model ready" if ready else "Training required"
    st.markdown(
        f"<div class='dashboard-heading'><div><h1>Character intelligence</h1>"
        f"<p>Your EMNIST model, evaluation, and recent recognition activity.</p></div>"
        f"<span class='status-pill'>● {status}</span></div>",
        unsafe_allow_html=True,
    )
    if not ready:
        st.warning("Model not trained. Follow the dataset and training steps in the project README to enable predictions.")

    metrics_row = st.columns(4)
    metrics_row[0].metric("Official test accuracy", f"{metrics['accuracy']:.2%}" if metrics else "Not evaluated")
    metrics_row[1].metric("Held-out test samples", f"{metrics['sample_count']:,}" if metrics else "Not evaluated")
    metrics_row[2].metric("Letter classes", "26 · A–Z")
    metrics_row[3].metric("Session predictions", len(records))

    left, right = st.columns([1.65, 1], gap="medium")
    with left, st.container(border=True):
        st.markdown("### Class recognition quality")
        st.caption("Per-letter F1 on the official held-out test split")
        if metrics:
            report = metrics["classification_report"]
            letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
            selected_letter = st.selectbox("Inspect letter", letters, key="overview_letter")
            profile = class_test_profile(metrics, selected_letter)
            f1_values = [report[letter]["f1-score"] for letter in letters]
            selection = ["Selected" if letter == selected_letter else "Other letters" for letter in letters]
            figure = px.bar(
                x=letters,
                y=f1_values,
                labels={"x": "Letter", "y": "Test F1"},
                range_y=[0, 1],
                color=selection,
                color_discrete_map={"Selected": "#477b4c", "Other letters": "#d7dfd0"},
            )
            figure.update_layout(
                height=285,
                margin=dict(l=6, r=6, t=8, b=8),
                showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            figure.update_yaxes(tickformat=".0%", gridcolor="#edf0e9")
            figure.update_traces(marker_line_width=0)
            st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})

            selected_metrics = st.columns(4)
            selected_metrics[0].metric(f"{selected_letter} precision", f"{profile['precision']:.1%}")
            selected_metrics[1].metric(f"{selected_letter} recall", f"{profile['recall']:.1%}")
            selected_metrics[2].metric(f"{selected_letter} F1", f"{profile['f1']:.1%}")
            selected_metrics[3].metric("Test examples", profile["support"])

            if profile["confusions"]:
                st.caption(f"When the true letter is {selected_letter}, the model most often predicts:")
                confusion_figure = px.bar(
                    profile["confusions"],
                    x="count",
                    y="predicted_letter",
                    orientation="h",
                    labels={"count": "Misclassified examples", "predicted_letter": "Predicted as"},
                    color_discrete_sequence=["#db7958"],
                )
                confusion_figure.update_layout(showlegend=False, yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(
                    confusion_figure,
                    use_container_width=True,
                    config={"displayModeBar": False},
                )
            else:
                st.caption(f"No misclassified {selected_letter} samples were recorded in the saved test report.")
        else:
            st.info("Per-class test results will appear after official evaluation.")

    with right, st.container(border=True):
        st.markdown("### Recent activity")
        st.caption("Predictions made in this session")
        if records:
            for record in reversed(records[-5:]):
                st.markdown(
                    f"**{record['character']}** · {record['source']} · {record['confidence']:.1%}<br>"
                    f"<small>{record['timestamp']}</small>",
                    unsafe_allow_html=True,
                )
                st.divider()
        else:
            st.caption("No predictions in this session yet.")

    st.markdown("### Model pipeline")
    outline = st.columns(4)
    for column, title, detail in zip(
        outline,
        ("28 × 28 input", "3 convolution blocks", "Global pooling", "26-way softmax"),
        ("Normalized grayscale", "Edges, strokes, curves", "Compact feature vector", "A through Z"),
    ):
        with column:
            st.markdown(f"<div class='outline-item'><b>{title}</b><br><span>{detail}</span></div>", unsafe_allow_html=True)
    if training:
        st.caption(f"Training completed: {len(training.get('accuracy', []))} epochs · seed 42 · best validation checkpoint")
    else:
        st.caption("Dataset: EMNIST Letters · 26 classes · official test split reserved for final evaluation")