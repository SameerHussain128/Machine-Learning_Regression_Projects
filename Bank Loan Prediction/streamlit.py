import streamlit as st
import numpy as np
import joblib

# Load the trained model and scaler
model = joblib.load('loan_prediction_model.pkl')
scaler = joblib.load('scaler.pkl')

# Function to make a prediction
def predict_loan_status(data):
    # Convert input data to numpy array
    input_data = np.array([data])

    # Standardize the input data
    std_data = scaler.transform(input_data)

    # Make a prediction
    prediction = model.predict(std_data)

    # Return the prediction result
    return 'Loan approved' if prediction[0] == 1 else 'Loan not approved'

# Streamlit interface
st.title('Loan Prediction App')

# Collect input data from user
gender = st.selectbox('Gender', [0, 1])
married = st.selectbox('Married', [0, 1])
dependents = st.selectbox('Dependents', [0, 1, 2, 3])
education = st.selectbox('Education', [0, 1])
self_employed = st.selectbox('Self Employed', [0, 1])
applicant_income = st.number_input('Applicant Income', min_value=0, value=0)
coapplicant_income = st.number_input('Coapplicant Income', min_value=0, value=0)
loan_amount = st.number_input('Loan Amount', min_value=0, value=0)
loan_amount_term = st.number_input('Loan Amount Term', min_value=0, value=360)
credit_history = st.selectbox('Credit History', [0, 1])
property_area = st.selectbox('Property Area', [0, 1, 2])

# Make prediction on button click
if st.button('Predict'):
    data = [
        gender,
        married,
        dependents,
        education,
        self_employed,
        applicant_income,
        coapplicant_income,
        loan_amount,
        loan_amount_term,
        credit_history,
        property_area
    ]
    
    result = predict_loan_status(data)
    st.write(result)
