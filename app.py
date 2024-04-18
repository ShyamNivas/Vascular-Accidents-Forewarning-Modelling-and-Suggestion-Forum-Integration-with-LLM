import streamlit as st
import pandas as pd
import pickle

# Load the saved model
model = pickle.load(open('random_forest_model.pkl', 'rb'))

# Streamlit UI setup
st.title('Brain Stroke Prediction')

# Collect user inputs for all necessary features
age = st.number_input('Age', value=75, min_value=0, step=1, format='%d')
avg_glucose_level = st.number_input('Average Glucose Level', value=300.0, min_value=0.0)
bmi = st.number_input('BMI', value=36.6, min_value=0.0)

gender_Male = st.radio('Gender', ['Male', 'Female']) == 'Male'
ever_married_Yes = st.radio('Ever Married', ['Yes', 'No']) == 'Yes'

work_type = st.radio('Work Type', ['Never worked', 'Private', 'Self-employed', 'Children'])
work_type_Never_worked = work_type == 'Never worked'
work_type_Private = work_type == 'Private'
work_type_Self_employed = work_type == 'Self-employed'
work_type_children = work_type == 'Children'

Residence_type_Urban = st.radio('Residence Type', ['Urban', 'Rural']) == 'Urban'

smoking_status = st.radio('Smoking Status', ['Formerly smoked', 'Never smoked', 'Smokes'])
smoking_status_formerly_smoked = smoking_status == 'Formerly smoked'
smoking_status_never_smoked = smoking_status == 'Never smoked'
smoking_status_smokes = smoking_status == 'Smokes'

hypertension_1 = st.radio('Hypertension', ['Yes', 'No']) == 'Yes'
heart_disease_1 = st.radio('Heart Disease', ['Yes', 'No']) == 'Yes'

# Prepare the input for prediction
if st.button('Predict Stroke Risk'):
    user_input = {
        'age': age,
        'avg_glucose_level': avg_glucose_level,
        'bmi': bmi,
        'gender_Male': int(gender_Male),
        'hypertension_1': int(hypertension_1),
        'heart_disease_1': int(heart_disease_1),
        'ever_married_Yes': int(ever_married_Yes),
        'work_type_Never_worked': int(work_type_Never_worked),
        'work_type_Private': int(work_type_Private),
        'work_type_Self-employed': int(work_type_Self_employed),  # Corrected feature name
        'work_type_children': int(work_type_children),
        'Residence_type_Urban': int(Residence_type_Urban),
        'smoking_status_formerly smoked': int(smoking_status_formerly_smoked),  # Corrected feature name
        'smoking_status_never smoked': int(smoking_status_never_smoked),  # Corrected feature name
        'smoking_status_smokes': int(smoking_status_smokes),
    }
    processed_input = pd.DataFrame([user_input])
    prediction = model.predict(processed_input)
    
    # Display prediction result
    if prediction[0] == 0:
        st.write('Low Risk of Stroke')
    else:
        st.write('High Risk of Stroke')







