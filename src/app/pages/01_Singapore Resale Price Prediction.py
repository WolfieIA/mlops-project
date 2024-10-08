import streamlit as st
import pandas as pd
import numpy as np
from pycaret.regression import load_model, predict_model
import math
from PIL import Image


st.set_page_config(page_title="Singapore Resale Price Prediction", page_icon="📈")

st.markdown("# Singapore Resale Price Prediction 🏠")
st.sidebar.header("Singapore Resale Price Prediction")
hdb_image = Image.open('src/statics/img/HDB.jpg')
st.image(hdb_image, width=700)
st.sidebar.info('This app is created to predict the housing price. Please adjust the features accordingly.')
st.sidebar.text('Done by: Azrul (223888Z)')
# Load the PyCaret model
model = load_model('models/best_model_regression')

def predict(data):
    # Convert the input data to a DataFrame
    df = pd.DataFrame([data])
    
    # Use PyCaret's predict_model function
    predictions = predict_model(model, data=df)
    
    # Get the logged prediction
    logged_prediction = predictions['prediction_label'].iloc[0]
    
    # Inverse transform the prediction (exp because we used log)
    actual_prediction = math.exp(logged_prediction)
    
    return actual_prediction

def main():
    st.info("Please fill in the details of the HDB flat to predict its resale price.",icon="ℹ️")

    # Input fields
    block = st.text_input("Block")
    street_name = st.text_input("Street Name")
    town = st.selectbox("Town", ["Ang Mo Kio", "Bedok", "Bishan", "Bukit Batok"])  # Add more towns as needed
    postal_code = st.text_input("Postal Code")
    flat_type = st.selectbox("Flat Type", ["1 Room", "2 Room", "3 Room", "4 Room", "5 Room", "Executive", "Multi-Generation"])
    storey_range = st.selectbox("Storey Range", ["01 to 03", "04 to 06", "07 to 09", "10 to 12", "13 to 15", "16 to 18", "19 to 21", "22 to 24", "25 to 27", "28 to 30"])
    flat_model = st.selectbox("Flat Model", ["Improved", "New Generation", "Model A", "Standard", "Apartment", "Simplified"])
    floor_area_sqm = st.number_input("Floor Area (sqm)", format="%.2f")
    cbd_dist = st.number_input("CBD Distance", format="%.2f")
    min_dist_mrt = st.number_input("Min Distance to MRT", format="%.2f")
    
    if st.button("Predict"):
        data = {
            'block': block,
            'street_name': street_name,
            'town': town,
            'postal_code': postal_code,
            'flat_type': flat_type,
            'storey_range': storey_range,
            'flat_model': flat_model,
            'floor_area_sqm': floor_area_sqm,
            'cbd_dist': cbd_dist,
            'min_dist_mrt': min_dist_mrt
        }

        try:
            prediction = predict(data)
            st.write(f"Predicted Resale Price: ${prediction:,.2f}")
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")
            st.error(f"Error details: {str(e.__class__.__name__)}: {str(e)}")

if __name__ == "__main__":
    main()