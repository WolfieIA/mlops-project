import streamlit as st



if "page_config" not in st.session_state:
    st.set_page_config(
        page_title="Machine Learning Operation Prediction App",
        page_icon="👋",
    )
    st.session_state.page_config = True

st.write("# Welcome to my Streamlit App! 👋")

st.sidebar.success("Select a page above.")

st.markdown(
    """
    This is the home page of our multi-page app.
    Select a page from the sidebar to explore different functionalities!
    """
)