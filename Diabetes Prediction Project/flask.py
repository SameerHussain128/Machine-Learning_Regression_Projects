from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Loading the saved model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))

# Creating a function for Prediction
def diabetes_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data, dtype=np.float64)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)
    if prediction[0] == 0:
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for the prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        input_data = [
            float(request.form['Pregnancies']),
            float(request.form['Glucose']),
            float(request.form['BloodPressure']),
            float(request.form['SkinThickness']),
            float(request.form['Insulin']),
            float(request.form['BMI']),
            float(request.form['DiabetesPedigreeFunction']),
            float(request.form['Age'])
        ]
        diagnosis = diabetes_prediction(input_data)
        return render_template('result.html', diagnosis=diagnosis)

if __name__ == '__main__':
    app.run(debug=True)