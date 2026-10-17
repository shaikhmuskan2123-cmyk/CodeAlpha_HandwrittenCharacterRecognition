"""Single-character image upload and export."""

import streamlit as st

from pages.common import result_downloads, show_predictions, validate_upload


def render(get_model) -> None:
    st.markdown("## Upload one character")
    st.caption("PNG or JPEG · one handwritten English letter per image · maximum 10 MB")
    uploaded = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"], key="single_upload")
    if uploaded is None:
        st.info("Words and multi-character images are not supported by this classifier.")
        return
    try:
        image = validate_upload(uploaded)
        st.image(image, caption=uploaded.name, width=360)
        if st.button("Predict letter", type="primary"):
            try:
                result = show_predictions(get_model(), image, "Upload", thumbnail=image)
                result_downloads(result, "glyphsense_prediction")
            except FileNotFoundError as error:
                st.error(str(error))
            except ValueError as error:
                st.info(str(error))
            except Exception as error:
                st.exception(error)
    except ValueError as error:
        st.error(str(error))