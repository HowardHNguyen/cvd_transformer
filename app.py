import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
from sklearn.metrics import accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt

# Load the scaler and model
scaler = joblib.load('scaler.pkl')
model = TransformerModel(input_dim=15, num_classes=2)
model.load_state_dict(torch.load('transformer_model.pth', map_location=torch.device('cpu')))
model.eval()

# Define features and target variable
feature_columns = ['AGE', 'TOTCHOL', 'SYSBP', 'DIABP', 'BMI', 'CURSMOKE', 'GLUCOSE', 'DIABETES', 'HEARTRTE', 'CIGPDAY', 'BPMEDS', 'STROKE', 'HYPERTEN', 'LDLC', 'HDLC']

# Streamlit app
st.title("Cardiovascular Disease Prediction (Transformer)")

st.sidebar.header('User Input Parameters')

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
    ldlc = st.sidebar.slider('LDL Cholesterol', 0, 200, 100)
    hdlc = st.sidebar.slider('HDL Cholesterol', 0, 100, 50)
    
    data = {
        'AGE': age,
        'TOTCHOL': totchol,
        'SYSBP': sysbp,
        'DIABP': diabp,
        'BMI': bmi,
        'CURSMOKE': cursmoke,
        'GLUCOSE': glucose,
        'DIABETES': diabetes,
        'HEARTRTE': heartrate,
        'CIGPDAY': cigpday,
        'BPMEDS': bpmeds,
        'STROKE': stroke,
        'HYPERTEN': hyperten,
        'LDLC': ldlc,
        'HDLC': hdlc
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

st.subheader('User Input Parameters')
st.write(input_df)

if st.button('PREDICT NOW'):
    # Preprocess the input data
    input_scaled = scaler.transform(input_df)

    # Convert to tensor
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32).unsqueeze(1)

    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.argmax(output, 1).item()
        prediction_proba = torch.softmax(output, dim=1).numpy()[0]

    # Display prediction
    st.subheader('Prediction')
    if prediction == 0:
        st.write('No Cardiovascular Disease')
    else:
        st.write('Cardiovascular Disease')

    st.subheader('Prediction Probability')
    st.write(pd.DataFrame(prediction_proba, columns=['Probability'], index=['No CVD', 'CVD']))

    # Load true labels and predicted probabilities for ROC curve
    y_test = joblib.load('transformer_y_test.pkl')
    y_pred_proba = joblib.load('transformer_y_pred_proba.pkl')

    # Compute ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    st.subheader('Model Performance (ROC Curve)')
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='blue', lw=2, label=f'Transformer (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc="lower right")
    st.pyplot(fig)

    # Feature importances (Placeholder for demonstration)
    st.subheader('Feature Importances')
    importances = np.random.rand(len(feature_columns))  # Placeholder importances
    indices = np.argsort(importances)

    fig, ax = plt.subplots()
    ax.barh(range(len(indices)), importances[indices], align='center')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_columns[i] for i in indices])
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importances (Transformer)')
    st.pyplot(fig)
