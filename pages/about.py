"""Beginner-friendly notes about the model and project."""

import streamlit as st


def render(get_model) -> None:
    st.markdown("## About GlyphSense AI")
    st.write(
        "Built for Task 3 of the CodeAlpha Machine Learning Internship, GlyphSense recognizes individual "
        "English alphabet characters with a convolutional neural network trained on EMNIST Letters."
    )
    with st.expander("How does a CNN recognize a letter?", expanded=True):
        st.write(
            "A convolution slides small learned filters across the image. Early filters respond to local edges "
            "and stroke directions; deeper layers combine those into corners, curves, and letter-like shapes."
        )
        st.write(
            "Max pooling reduces the spatial grid while retaining strong responses. Global pooling and dense layers "
            "combine those features, then softmax scores the 26 possible classes."
        )
    with st.expander("What does the score mean?"):
        st.write(
            "The displayed score is the model's softmax output for that class. Neural networks can be overconfident, "
            "so this score is not automatically a calibrated probability that the prediction is correct."
        )
    with st.expander("Data, limitations, and common mistakes"):
        st.write(
            "Training and evaluation use the official EMNIST Letters train and held-out test splits. EMNIST contains "
            "small grayscale characters; user images are cropped, resized, centered, and normalized to match that format."
        )
        st.write(
            "Only one character is classified at a time. The model does not recognize words, handwriting lines, "
            "punctuation, or non-English scripts. Similar shapes such as C/G, I/J, and O/Q can be confused, especially "
            "when writing differs from the dataset. No Grad-CAM visualization is provided by this compact classifier."
        )
    st.caption("Dataset: NIST EMNIST Letters · 26 classes · official test split reserved for final evaluation")