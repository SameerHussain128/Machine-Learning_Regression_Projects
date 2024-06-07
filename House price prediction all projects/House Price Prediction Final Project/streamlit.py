import streamlit as st
import pandas as pd
import joblib

# Load the pre-trained model
model_rf = joblib.load('model_rf.pkl')

# Title of the app
st.title('House Price Prediction')

# User inputs for the model
KitchenAbvGr = st.number_input('Number of Kitchens Above Ground', min_value=0, max_value=10, value=2)
TotRmsAbvGrd = st.number_input('Total Rooms Above Ground', min_value=0, max_value=20, value=8)
PoolArea = st.number_input('Pool Area', min_value=0, max_value=1000, value=500)
LotArea = st.number_input('Lot Area', min_value=0, max_value=100000, value=9000)
GrLivArea = st.number_input('Above Ground Living Area', min_value=0, max_value=10000, value=7000)
OverallQual = st.number_input('Overall Quality', min_value=1, max_value=10, value=7)
TotalBath = st.number_input('Total Bathrooms', min_value=0, max_value=10, value=5)
BedroomAbvGr = st.number_input('Bedrooms Above Ground', min_value=0, max_value=20, value=8)
GarageArea = st.number_input('Garage Area', min_value=0, max_value=1000, value=600)
TotalSF = st.number_input('Total Square Footage', min_value=0, max_value=10000, value=4000)
YearBuilt = st.number_input('Year Built', min_value=1800, max_value=2024, value=2009)
YrSold = st.number_input('Year Sold', min_value=1900, max_value=2024, value=2003)

# Create a DataFrame from user inputs
input_data = {
    'KitchenAbvGr': [KitchenAbvGr],
    'TotRmsAbvGrd': [TotRmsAbvGrd],
    'PoolArea': [PoolArea],
    'LotArea': [LotArea],
    'GrLivArea': [GrLivArea],
    'OverallQual': [OverallQual],
    'TotalBath': [TotalBath],
    'BedroomAbvGr': [BedroomAbvGr],
    'GarageArea': [GarageArea],
    'TotalSF': [TotalSF],
    'YearBuilt': [YearBuilt],
    'YrSold': [YrSold]

}

new_data_point = pd.DataFrame(input_data)

# Button for Prediction
if st.button('Predict House Price'):
    # Predict the price using the RandomForest model
    predicted_price = model_rf.predict(new_data_point)
    st.write(f"Predicted Price for the new data point: ${predicted_price[0]:,.2f}")

