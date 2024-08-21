import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Load the trained logistic regression model
log_reg_model = joblib.load('model/logistic_regression_model_8_cat2.pkl')
# Load the scalers and encoders
scaler = joblib.load('model/scaler_cat2.pkl')
label_encoders = joblib.load('model/label_encoders_cat2.pkl')

# Streamlit form
st.header("Modelo predictivo #2")
st.write("Probabilidad de que al menos 8 folículos lleguen a Metafase II")

with st.form("input_form"):
    age = st.number_input("Edad paciente", min_value=15, max_value=70, value=30)
    options_map = {
        'DO': 'Donación de ovocitos y ovodonación',
        'FM/ICSI': 'Factor masculino/ICSI',
        'VO': 'Vitrificación de ovocitos',
        'BRO/ESCA': 'Baja reserva ovárica/Esterilidad sin causa aparente',
        'ERA': 'Edad reproductiva avanzada',
        'otro': 'Otro'
    }

    # Display the selectbox with long names
    selected_long_name = st.selectbox("Motivo de consulta", list(options_map.values()))
    # cat_motivo = st.selectbox("Motivo de consulta", ['DO ', 'FM/ICSI', 'VO ', 'BRO/ESCA', 'ERA', 'otro'])
    dias_estimulo = st.number_input("Días de estímulo", min_value=0.0, value=0.0)
    descarga_ovul = st.selectbox("Descarga de ovulación", ['GnRHa', 'hCG'])

    submit_button = st.form_submit_button(label="Predecir")

if submit_button:
    # Prepare the input data
    # Reverse the map to get short names based on selected long descriptions
    reverse_map = {v: k for k, v in options_map.items()}
    # Retrieve the corresponding short name
    cat_motivo = reverse_map[selected_long_name]
    input_data = pd.DataFrame({
        'edad_paciente': [age],
        'Cat_Motivo': [cat_motivo],
        'dias_estimulo': [dias_estimulo],
        'descarga_ovul': [descarga_ovul],
    })
    # st.write(input_data)

    
    # Scale numeric columns
    numeric_cols = ['edad_paciente', 'dias_estimulo']
    input_data[numeric_cols] = scaler.transform(input_data[numeric_cols])

    # Encode non-numeric columns
    for col in ['Cat_Motivo', 'descarga_ovul']:
        input_data[col] = label_encoders[col].transform(input_data[col])

    input_data = input_data[['edad_paciente', 'dias_estimulo', 'Cat_Motivo', 'descarga_ovul']]    
    # st.write(input_data)
    # Predict probabilities
    prediction = log_reg_model.predict(input_data)
    probability = log_reg_model.predict_proba(input_data)

    # Display results
    st.write(f"Predicción: {'Positiva' if prediction[0] else 'Negativa'}")
    st.write(probability)
    st.write(f"Hay un {probability[0][1] * 100:.2f}% de probabilidad de que 8 o más folículos lleguen a Metafase II.")

