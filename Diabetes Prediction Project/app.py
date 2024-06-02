import numpy as np
import pickle
import streamlit as st

# Loading the saved model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))

# Creating a function for Prediction
def diabetes_prediction(input_data):
    # Changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data, dtype=np.float64)

    # Reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # Make a prediction
    prediction = loaded_model.predict(input_data_reshaped)

    # Return the result
    if prediction[0] == 0:
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'

# Main function for the Streamlit app
def main():
    # Giving a title
    st.title('Diabetes Prediction Web App')

    # Getting the input data from the user
    Pregnancies = st.text_input('Number of Pregnancies', '0')
    Glucose = st.text_input('Glucose Level', '0')
    BloodPressure = st.text_input('Blood Pressure value', '0')
    SkinThickness = st.text_input('Skin Thickness value', '0')
    Insulin = st.text_input('Insulin Level', '0')
    BMI = st.text_input('BMI value', '0')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value', '0.0')
    Age = st.text_input('Age of the Person', '0')

    # Code for Prediction
    diagnosis = ''

    # Creating a button for Prediction
    if st.button('Diabetes Test Result'):
        try:
            input_data = [
                float(Pregnancies),
                float(Glucose),
                float(BloodPressure),
                float(SkinThickness),
                float(Insulin),
                float(BMI),
                float(DiabetesPedigreeFunction),
                float(Age)
            ]
            diagnosis = diabetes_prediction(input_data)
        except ValueError:
            diagnosis = 'Please enter valid input values.'

    st.success(diagnosis)

if __name__ == '__main__':
    main()
