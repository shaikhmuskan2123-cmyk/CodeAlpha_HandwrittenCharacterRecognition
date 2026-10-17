"""Interactive sketchpad and single-character prediction."""

import streamlit as st

from pages.common import show_predictions


def render(get_model) -> None:
    st.markdown("## Draw a letter")
    st.caption("Sketch one uppercase English letter. Predictions come from the trained EMNIST CNN.")
    if not st.session_state.get("model_ready", True):
        st.error("The trained model is not available. Complete the training steps in README.md first.")
        return
    try:
        from streamlit_drawable_canvas import st_canvas
    except ImportError:
        st.error("Drawing support is not installed. Install project dependencies with `pip install -r requirements.txt`.")
        return

    if "canvas_version" not in st.session_state:
        st.session_state["canvas_version"] = 0
    if st.button("Clear canvas"):
        st.session_state["canvas_version"] += 1
        st.rerun()

    brush = st.slider("Brush thickness", min_value=8, max_value=32, value=18, step=2)
    canvas = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=brush,
        stroke_color="#101821",
        background_color="#ffffff",
        height=320,
        width=320,
        drawing_mode="freedraw",
        key=f"letter_canvas_{st.session_state['canvas_version']}",
    )
    predict_button = st.button("Predict letter", type="primary", disabled=canvas.image_data is None)
    if predict_button:
        try:
            with st.spinner("Reading the strokes…"):
                model = get_model()
                show_predictions(model, canvas.image_data, "Drawing")
        except FileNotFoundError as error:
            st.error(str(error))
        except ValueError as error:
            st.info(str(error))
        except Exception as error:
            st.exception(error)