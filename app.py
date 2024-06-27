import streamlit as st
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

# Load the model and scaler
transformer_model = TransformerModel(input_dim=15, num_classes=2)
transformer_model.load_state_dict(torch.load('transformer_model.pth', map_location=torch.device('cpu')))
transformer_model.eval()
scaler = joblib.load('scaler.pkl')

# Function to compute SHAP values
def compute_shap_values(model, input_tensor):
    # Use DeepExplainer for the transformer model
    explainer = shap.DeepExplainer(model, input_tensor)
    shap_values = explainer.shap_values(input_tensor)
    return shap_values

# Sidebar for user input parameters
st.sidebar.header('Please Select Your Parameters')
def user_input_features():
    age = st.sidebar.slider('Age', 32, 81, 54)
    totchol = st.sidebar.slider('Total Cholesterol', 107, 696, 200)
    sysbp = st.sidebar.slider('Systolic Blood Pressure', 83, 295, 140)
    diabp = st.sidebar.slider('Diastolic Blood Pressure', 30, 150, 89)
    bmi = st.sidebar.slider('BMI', 14.43, 56.80, 26.77)
    cursmoke = st.sidebar.selectbox('Current Smoker', [0, 1])
    glucose = st.sidebar.slider('Glucose', 39, 478, 117)
    diabetes = st.sidebar.selectbox('Diabetes', [0, 1])
    heartrate = st.sidebar.slider('Heart Rate', 37, 220, 91)
    cigpday = st.sidebar.slider('Cigarettes Per Day', 0, 90, 20)
    bpmeds = st.sidebar.selectbox('On BP Meds', [0, 1])
    stroke = st.sidebar.selectbox('Stroke', [0, 1])
    hyperten = st.sidebar.selectbox('Hypertension', [0, 1])
    ldlc = st.sidebar.slider('LDL Cholesterol', 0, 208, 100)
    hdlc = st.sidebar.slider('HDL Cholesterol', 0, 100, 50)
    data = {'AGE': age, 'TOTCHOL': totchol, 'SYSBP': sysbp, 'DIABP': diabp, 'BMI': bmi,
            'CURSMOKE': cursmoke, 'GLUCOSE': glucose, 'DIABETES': diabetes, 'HEARTRTE': heartrate,
            'CIGPDAY': cigpday, 'BPMEDS': bpmeds, 'STROKE': stroke, 'HYPERTEN': hyperten,
            'LDLC': ldlc, 'HDLC': hdlc}
    features = pd.DataFrame(data, index=[0])
    return features

input_data = user_input_features()

# Apply scaling
input_data_scaled = scaler.transform(input_data)
input_tensor = torch.tensor(input_data_scaled, dtype=torch.float32)

# Prediction button
if st.sidebar.button("PREDICT NOW"):
    with torch.no_grad():
        output = transformer_model(input_tensor.unsqueeze(1))
        _, prediction = torch.max(output, 1)
        prediction_proba = F.softmax(output, dim=1).numpy()

    st.subheader('Prediction')
    if prediction.item() == 1:
        st.write("Cardiovascular Disease Detected")
    else:
        st.write("No Cardiovascular Disease")

    st.subheader('Prediction Probability')
    proba_df = pd.DataFrame(prediction_proba, columns=['No CVD', 'CVD'])
    st.write(proba_df)
    fig, ax = plt.subplots()
    ax.bar(proba_df.columns, proba_df.iloc[0])
    ax.set_xlabel('Prediction')
    ax.set_ylabel('Probability')
    st.pyplot(fig)

    # Model Performance (ROC Curve)
    st.subheader('Model Performance (ROC Curve)')
    st.image('roc_curve.png')

    # Feature Importances using SHAP
    st.subheader("Feature Importances (Transformer)")
    shap_values = compute_shap_values(transformer_model, input_tensor)
    shap_values = shap_values[0]  # For classification, shap_values is a list
    shap.summary_plot(shap_values, input_data, plot_type="bar", feature_names=input_data.columns.tolist())
    st.pyplot(bbox_inches='tight')
