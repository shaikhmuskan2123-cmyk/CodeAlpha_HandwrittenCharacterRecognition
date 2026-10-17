"""Visualize only saved training and official held-out evaluation artifacts."""

import numpy as np
import plotly.express as px
import streamlit as st

from config import CONFIG_PATH, HISTORY_PATH, METRICS_PATH, MODEL_PATH, REPORT_DIR
from src.dashboard_data import class_test_profile
from src.utils import load_json


def _chart_layout(figure, height: int = 350):
    figure.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=10, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#3c443b",
    )
    return figure


def render(get_model) -> None:
    st.markdown("## Model analytics")
    st.caption("Evaluation uses the official held-out EMNIST test split. Training metrics are not substituted here.")
    metrics = load_json(METRICS_PATH)
    if not metrics:
        st.info("No evaluation metrics are available yet. Train the model and run `python -m scripts.evaluate_model`.")
        return

    columns = st.columns(4)
    columns[0].metric("Test accuracy", f"{metrics['accuracy']:.2%}")
    columns[1].metric("Macro precision", f"{metrics['precision_macro']:.2%}")
    columns[2].metric("Macro recall", f"{metrics['recall_macro']:.2%}")
    columns[3].metric("Macro F1", f"{metrics['f1_macro']:.2%}")
    st.caption(f"{metrics['sample_count']:,} official test images · 800 examples per letter")

    report = metrics["classification_report"]
    class_rows = [
        {"Letter": letter, "Precision": report[letter]["precision"], "Recall": report[letter]["recall"], "F1": report[letter]["f1-score"]}
        for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    ]
    strongest = max(class_rows, key=lambda row: row["F1"])
    weakest = min(class_rows, key=lambda row: row["F1"])
    insights = st.columns(2)
    insights[0].metric("Strongest class", strongest["Letter"], f"F1 {strongest['F1']:.1%}")
    insights[1].metric("Needs most attention", weakest["Letter"], f"F1 {weakest['F1']:.1%}")

    selected_letter = st.selectbox("Explore test results by letter", list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), key="analytics_letter")
    profile = class_test_profile(metrics, selected_letter)
    selected_metrics = st.columns(4)
    selected_metrics[0].metric(f"{selected_letter} precision", f"{profile['precision']:.1%}")
    selected_metrics[1].metric(f"{selected_letter} recall", f"{profile['recall']:.1%}")
    selected_metrics[2].metric(f"{selected_letter} F1", f"{profile['f1']:.1%}")
    selected_metrics[3].metric("Test examples", profile["support"])
    if profile["confusions"]:
        st.caption(f"True {selected_letter} examples most often misclassified as:")
        confusion_figure = px.bar(
            profile["confusions"],
            x="count",
            y="predicted_letter",
            orientation="h",
            labels={"count": "Misclassified examples", "predicted_letter": "Predicted as"},
            color_discrete_sequence=["#db7958"],
        )
        confusion_figure.update_layout(showlegend=False, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(confusion_figure, use_container_width=True, config={"displayModeBar": False})
    else:
        st.caption(f"No misclassified {selected_letter} samples were recorded in the saved test report.")

    left, right = st.columns([1.05, 1.4], gap="medium")
    with left:
        st.markdown("### Per-letter quality")
        letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        f1_values = [report[letter]["f1-score"] for letter in letters]
        selection = ["Selected" if letter == selected_letter else "Other letters" for letter in letters]
        figure = px.bar(
            x=letters,
            y=f1_values,
            labels={"x": "Letter", "y": "F1 score"},
            range_y=[0, 1],
            color=selection,
            color_discrete_map={"Selected": "#477b4c", "Other letters": "#d7dfd0"},
        )
        figure.update_yaxes(tickformat=".0%", gridcolor="#e6e9e1")
        figure.update_layout(showlegend=False)
        figure.update_traces(marker_line_width=0)
        st.plotly_chart(_chart_layout(figure, 370), use_container_width=True, config={"displayModeBar": False})

    with right:
        st.markdown("### Confusion matrix · row-normalized")
        matrix = np.asarray(metrics["confusion_matrix"], dtype=np.float64)
        row_totals = matrix.sum(axis=1, keepdims=True)
        normalized = np.divide(matrix, row_totals, out=np.zeros_like(matrix), where=row_totals != 0)
        figure = px.imshow(
            normalized,
            x=list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
            y=list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
            labels={"x": "Predicted letter", "y": "True letter", "color": "Recall"},
            color_continuous_scale=[[0, "#f0f2eb"], [0.45, "#d8e986"], [1, "#477b4c"]],
            aspect="auto",
            zmin=0,
            zmax=1,
        )
        figure.update_traces(hovertemplate="True %{y} · predicted %{x}<br>Recall %{z:.1%}<extra></extra>")
        st.plotly_chart(_chart_layout(figure, 490), use_container_width=True, config={"displayModeBar": False})

    st.markdown("### Most frequent confusions")
    matrix_counts = np.asarray(metrics["confusion_matrix"])
    off_diagonal = matrix_counts.copy()
    np.fill_diagonal(off_diagonal, 0)
    indices = np.dstack(np.unravel_index(np.argsort(off_diagonal.ravel())[::-1], off_diagonal.shape))[0]
    confusion_rows = []
    for true_index, predicted_index in indices:
        count = int(off_diagonal[true_index, predicted_index])
        if count == 0 or len(confusion_rows) == 8:
            break
        confusion_rows.append(
            {
                "True letter": chr(65 + int(true_index)),
                "Predicted as": chr(65 + int(predicted_index)),
                "Errors": count,
                "Share of true class": count / max(1, int(matrix_counts[true_index].sum())),
            }
        )
    st.dataframe(confusion_rows, use_container_width=True, hide_index=True)

    st.markdown("### Per-class performance")
    st.dataframe(class_rows, use_container_width=True, hide_index=True)

    training_history = load_json(HISTORY_PATH)
    if training_history:
        st.markdown("### Training history")
        history_rows = [
            {"Epoch": index + 1, "Training accuracy": acc, "Validation accuracy": training_history["val_accuracy"][index]}
            for index, acc in enumerate(training_history.get("accuracy", []))
        ]
        accuracy_figure = px.line(
            history_rows,
            x="Epoch",
            y=["Training accuracy", "Validation accuracy"],
            markers=True,
            color_discrete_sequence=["#477b4c", "#db7958"],
        )
        accuracy_figure.update_yaxes(range=[0, 1], tickformat=".0%", gridcolor="#e6e9e1")
        st.plotly_chart(_chart_layout(accuracy_figure, 300), use_container_width=True, config={"displayModeBar": False})
        loss_rows = [
            {"Epoch": index + 1, "Training loss": loss, "Validation loss": training_history["val_loss"][index]}
            for index, loss in enumerate(training_history.get("loss", []))
        ]
        loss_figure = px.line(
            loss_rows,
            x="Epoch",
            y=["Training loss", "Validation loss"],
            markers=True,
            color_discrete_sequence=["#477b4c", "#db7958"],
        )
        loss_figure.update_yaxes(gridcolor="#e6e9e1")
        st.plotly_chart(_chart_layout(loss_figure, 300), use_container_width=True, config={"displayModeBar": False})

    examples_path = REPORT_DIR / metrics.get("misclassified_examples", "")
    if examples_path.is_file():
        examples = np.load(examples_path)
        st.markdown("### Held-out misclassifications")
        if len(examples["images"]):
            cards = st.columns(5)
            for index, column in enumerate(cards):
                if index >= len(examples["images"]):
                    break
                true_label = chr(65 + int(examples["true_labels"][index]))
                predicted_label = chr(65 + int(examples["predicted_labels"][index]))
                column.image(examples["images"][index], clamp=True)
                column.caption(f"True {true_label} · predicted {predicted_label}")
        else:
            st.write("No misclassifications in the stored sample.")

    with st.expander("Saved model configuration and architecture"):
        config = load_json(CONFIG_PATH)
        st.json(config or {"model_file": str(MODEL_PATH.name)})
        if MODEL_PATH.is_file():
            try:
                lines = []
                get_model().summary(print_fn=lines.append)
                st.code("\n".join(lines))
            except Exception as error:
                st.caption(f"Model summary unavailable: {error}")
    st.caption("Softmax scores are not calibrated probabilities of correctness.")