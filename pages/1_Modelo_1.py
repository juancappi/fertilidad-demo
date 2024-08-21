import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Load the trained logistic regression model
log_reg_model = joblib.load('model/logistic_regression_model_8_s_fa.pkl')
# Load the scalers and encoders
scaler = joblib.load('model/scaler_s_fa.pkl')
label_encoders = joblib.load('model/label_encoders_s_fa.pkl')

# Streamlit form
st.header("Modelo predictivo #1")
st.write("Probabilidad de que al menos 8 folículos lleguen a Metafase II")

with st.form("input_form"):
    age = st.number_input("Edad paciente", min_value=15, max_value=70, value=30)
    IMC = st.number_input("Indice de masa corporal", min_value=0.0, value=0.0)
    CFA = st.number_input("Conteo folicular antral", min_value=0.0, value=0.0)
    HAM = st.number_input("HAM", min_value=0.0, value=0.0)
    
    prev_child = st.checkbox("Tuvo hijos anteriormente?")
    smokes = st.selectbox("Fuma?", ["si", "no", "ex fumadora"])

    submit_button = st.form_submit_button(label="Predecir")

if submit_button:
    # Prepare the input data
    input_data = pd.DataFrame({
        '9 Edad paciente': [age],
        'IMC': [IMC],
        '25 CFA': [CFA],
        '26 HAM': [HAM],
        'Hijos Previos': ['si' if prev_child else 'no'],
        'Fuma': [smokes]
    })
    
    # Scale numeric columns
    numeric_cols = ['9 Edad paciente', 'IMC', '25 CFA', '26 HAM']
    input_data[numeric_cols] = scaler.transform(input_data[numeric_cols])

    # Encode non-numeric columns
    for col in ['Hijos Previos', 'Fuma']:
        input_data[col] = label_encoders[col].transform(input_data[col])

    input_data = input_data[['9 Edad paciente', 'IMC', '25 CFA', '26 HAM', 'Hijos Previos', 'Fuma']]    
    # Predict probabilities
    prediction = log_reg_model.predict(input_data)
    probability = log_reg_model.predict_proba(input_data)

    # Display results
    st.write(f"Predicción: {'Positiva' if prediction[0] else 'Negativa'}")
    st.write(probability)
    st.write(f"Hay un {probability[0][1] * 100:.2f}% de probabilidad de que 8 o más folículos lleguen a Metafase II.")

