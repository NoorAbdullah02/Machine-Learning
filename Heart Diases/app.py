import streamlit as st
import pandas as pd
import joblib

model = joblib.load('KNN_heart.pkl')
scaler = joblib.load('scaler.pkl')
expected_columns = joblib.load('columns.pkl')

st.title('Heart Disease Prediction. By Noor Abdullah')

st.markdown('Please fill in the following details to predict the likelihood of heart disease:')
st.markdown('---')


age = st.number_input('Age', min_value=1, max_value=120, value=30)

sex = st.selectbox('Sex', ['Male', 'Female'])


chest_pain = st.selectbox(
    'Chest Pain Type',
    ['ATA', 'NAP', 'ASY', 'TA']
)


resting_bp = st.number_input('Resting Blood Pressure (mm Hg)', 
                             min_value=1, max_value=300, value=120)
cholesterol = st.number_input('Serum Cholesterol (mg/dl)', 
                                min_value=1, max_value=1000, value=200)

fasting_bs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['Yes', 'No'])

resting_ecg = st.selectbox(
    'Resting ECG Results',
    ['Normal', 'ST', 'LVH']
)   


max_hr = st.slider('Maximum Heart Rate Achieved', min_value=1, max_value=250, value=150)

exercise_angina = st.selectbox('Exercise Induced Angina', ['Yes', 'No'])

oldpeak = st.number_input('Oldpeak (ST depression)', min_value=0.0, max_value=10.0, value=1.0)

st_slope = st.selectbox(
    'ST Slope',
    ['Up', 'Flat']
)   



if st.button("Predict"):
    raw_input = {col: 0 for col in expected_columns}
    
    raw_input['Age'] = age
    raw_input['RestingBP'] = resting_bp
    raw_input['Cholesterol'] = cholesterol
    raw_input['MaxHR'] = max_hr
    raw_input['Oldpeak'] = oldpeak
    
    raw_input['FastingBS'] = 1 if fasting_bs == 'Yes' else 0
    raw_input['ExerciseAngina_Y'] = 1 if exercise_angina == 'Yes' else 0
    
    if sex == 'Male':
        raw_input['Sex_M'] = 1
    
    raw_input['ChestPainType_' + chest_pain] = 1
    
    raw_input['RestingECG_' + resting_ecg] = 1
    
    raw_input['ST_Slope_' + st_slope] = 1

    st.write("**Expected Columns:**")
    st.write(expected_columns)
    
    st.write("**Raw Input Dict:**")
    st.write(raw_input)
    
    input_df = pd.DataFrame([raw_input])
    input_df = input_df[expected_columns]

    for col in input_df.columns:
        input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
    input_df = input_df.fillna(0)
    
    st.write("**Input DataFrame:**")
    st.write(input_df)
    
    scaled_input = scaler.transform(input_df)
    
    st.write("**Scaled Input:**")
    st.write(scaled_input)
    
    prediction = model.predict(scaled_input)[0]
    
    st.write(f"**Prediction Value: {prediction}**")

    if prediction == 1:
        st.error('Prediction: High likelihood of heart disease. Please consult a doctor.')
    else:        
        st.success('Prediction: Low likelihood of heart disease. Keep up the healthy lifestyle!')