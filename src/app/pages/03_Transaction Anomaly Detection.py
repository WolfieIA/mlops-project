import streamlit as st
import pandas as pd
import pickle
from pycaret.anomaly import  predict_model, load_model
from PIL import Image
st.set_page_config(page_title="Transaction", page_icon="📊") 

st.markdown("# Transaction Anomaly Detection 📊") 
st.sidebar.header("Transaction Anomaly Detection") 
transaction = Image.open('src/statics/img/transaction.jpg')
st.image(transaction, width=700)
st.sidebar.info('This app is created to predict the transaction wheather is anormaly or not. Please adjust the features accordingly.')
st.sidebar.text('Done by: Raphael (210887Y)')

# Load the PyCaret model
model = load_model('models/iforest_anomaly_detection_pipeline')


def preprocess_input(data):
    # Convert the input data to a DataFrame
    df = pd.DataFrame([data])
    
    # Adjust the fiscal year information: Fiscal Year starts from 1 June
    df['FISCAL_YEAR_ADJ'] = df['FISCAL_YR'].astype(int)
    
    # Apply the same logic used during training if necessary
    df['FISCAL_YEAR_ADJ'] = df.apply(
        lambda row: row['FISCAL_YEAR_ADJ'] if int(row['FISCAL_MTH']) >= 6 else row['FISCAL_YEAR_ADJ'] - 1, axis=1
    )
    
    return df

def predict(data):
    # Preprocess the input data
    df = preprocess_input(data)
    
    # Use PyCaret's predict_model function
    predictions = predict_model(model, data=df)
    
    # Extract the anomaly prediction
    anomaly_label = predictions['Anomaly'].iloc[0]
    anomaly_score = predictions['Anomaly_Score'].iloc[0]
    
    return anomaly_label, anomaly_score

def main():
    st.info("Please fill in the details of the transaction to predict if it is an anomaly.",icon="ℹ️")

    # Create input fields for user data
    fiscal_year = st.text_input("Fiscal Year")
    fiscal_month = st.text_input("Fiscal Month")
    department = st.text_input("Department Name")
    division = st.text_input("Division Name")
    merchant = st.text_input("Merchant")
    category = st.text_input("Category Description")
    transaction_date = st.text_input("Transaction Date (YYYY-MM-DD)")
    amount = st.number_input("Transaction Amount", format="%.2f")

    if st.button("Predict Anomaly"):
        # Collect the input data into a dictionary
        data = {
            'FISCAL_YR': fiscal_year,
            'FISCAL_MTH': fiscal_month,
            'DEPT_NAME': department,
            'DIV_NAME': division,
            'MERCHANT': merchant,
            'CAT_DESC': category,
            'TRANS_DT': transaction_date,
            'AMT': amount
        }
        try:
            # Get the anomaly prediction and score
            anomaly_label, anomaly_score = predict(data)
            if anomaly_label == 1:
                st.write(f"Anomaly Detected with a score of {anomaly_score:.4f}")
            else:
                st.write(f"No Anomaly Detected (Score: {anomaly_score:.4f})")
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")
            st.error(f"Error details: {str(e.__class__.__name__)}: {str(e)}")

if __name__ == "__main__":
    main()
