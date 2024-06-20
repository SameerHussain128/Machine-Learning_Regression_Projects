from flask import Flask, request, render_template
import numpy as np
import joblib

app = Flask(__name__)

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

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        gender = int(request.form['gender'])
        married = int(request.form['married'])
        dependents = int(request.form['dependents'])
        education = int(request.form['education'])
        self_employed = int(request.form['self_employed'])
        applicant_income = float(request.form['applicant_income'])
        coapplicant_income = float(request.form['coapplicant_income'])
        loan_amount = float(request.form['loan_amount'])
        loan_amount_term = float(request.form['loan_amount_term'])
        credit_history = int(request.form['credit_history'])
        property_area = int(request.form['property_area'])

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
        return render_template('index.html', result=result)

    return render_template('index.html', result='')

if __name__ == '__main__':
    app.run(debug=True)
