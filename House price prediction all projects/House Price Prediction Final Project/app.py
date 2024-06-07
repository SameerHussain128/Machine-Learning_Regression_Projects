from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load the pre-trained Random Forest model
model_rf = joblib.load('model_rf.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_price():
    # Get user input from request
    data = request.get_json()

    # List of required features
    required_features = [
        'KitchenAbvGr', 'TotRmsAbvGrd', 'PoolArea', 'LotArea', 
        'GrLivArea', 'OverallQual', 'TotalBath', 'BedroomAbvGr', 
        'GarageArea', 'TotalSF', 'YearBuilt', 'YrSold'
    ]

    # Check if all required features are present in the input data
    if not all(feature in data for feature in required_features):
        return jsonify({'error': 'Missing required features'}), 400

    # Create DataFrame from user input
    input_data = pd.DataFrame([data])

    # Preprocess data if needed (scaling, etc.)
    # ... your data preprocessing logic here ...

    # Make prediction
    try:
        predicted_price = model_rf.predict(input_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    # Return prediction result
    return jsonify({'predicted_price': predicted_price[0]})

if __name__ == '__main__':
    app.run(debug=True)  # Run Flask app in debug mode
