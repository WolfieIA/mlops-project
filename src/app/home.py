import streamlit as st
from PIL import Image

def home():
    st.set_page_config(page_title="Multi-Model Predictor", page_icon="📊", layout="wide")

    # Title and introduction
    st.title("Welcome to my Machine Learning Operation App! 👋")
    st.markdown("""
    This is the home page of our multi-page app. Select a page from the sidebar to explore different functionalities!
    """)

    # Main content
    st.header("Available Prediction Models")

    # Create three columns for the different models
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Singapore Resale Price Prediction")
        st.image("src/statics/img/HDB.jpg", use_column_width=True)
        st.write("Predict housing prices in Singapore based on various features.")
        if st.button("Go to Singapore Resale Price Prediction"):
            st.switch_page("pages/01_residential.py")

    with col2:
        st.subheader("Mushroom Edibility Prediction")
        st.image("src/statics/img/mushroom_header.png", use_column_width=True)
        st.write("Determine if a mushroom is edible or poisonous based on its characteristics.")
        if st.button("Go to Mushroom Edibility Prediction"):
            st.switch_page("pages/02_mushroom_species.py")

    with col3:
        st.subheader("Transaction Anomaly Detection")
        st.image("src/statics/img/transaction.jpg", use_column_width=True)
        st.write("Detect anomalies in financial transactions to prevent fraud.")
        if st.button("Go to Transaction Anomaly Detection"):
            st.switch_page("pages/03_transaction.py")

    # Additional information
    st.header("About This App")
    st.write("""
    This multi-model predictor app showcases different machine learning models for various prediction tasks. 
    Each page focuses on a specific prediction problem and allows users to interact with the model by inputting 
    relevant features and receiving predictions.
    """)
    st.markdown("---")

    # Team information
    st.header("Our Team")
    st.write("This app was created by:")
    st.write("- Azrul (223888Z) Leader")
    st.write("- Lan (221528B)")
    st.write("- Raphael (210887Y)")

    
    st.markdown("---")

if __name__ == "__main__":
    home()