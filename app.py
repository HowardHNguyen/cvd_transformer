import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Define the Transformer model
class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_classes, d_model=128, max_seq_length=1, nhead=8, num_layers=3):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_seq_length, d_model), requires_grad=False)
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=512)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc1 = nn.Linear(d_model, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.max_seq_length = max_seq_length
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.embedding(x)
        batch_size, seq_length, _ = x.size()
        if seq_length > self.max_seq_length:
            raise ValueError(f"Input sequence length ({seq_length}) exceeds the maximum sequence length ({self.max_seq_length}).")
        pos_encoding = self.pos_encoder[:, :seq_length, :].expand(batch_size, -1, -1).to(x.device)
        x = x + pos_encoding
        x = x.transpose(0, 1)
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Load the model
transformer_model = TransformerModel(input_dim=15, num_classes=2)  # Adjust input_dim based on your dataset
transformer_model.load_state_dict(torch.load('transformer_model.pth', map_location=torch.device('cpu')))
transformer_model.eval()

# Load the scaler
scaler = joblib.load('scaler.pkl')

st.title('Cardiovascular Disease Prediction (Transformer)')

# User input
st.sidebar.header('User Input Parameters')

def user_input_features():
    age = st.sidebar.slider('Age', 32, 81, 54)
    totchol = st.sidebar.slider('Total Cholesterol', 107, 696, 200)
    sysbp = st.sidebar.slider('Systolic Blood Pressure', 83, 295, 140)
    diabp = st.sidebar.slider('Diastolic Blood Pressure', 30, 150, 89)
    bmi = st.sidebar.slider('BMI', 14.43, 56.80, 26.77)
    cursmoke = st.sidebar.slider('Current Smoker', 0, 1, 0)
    glucose = st.sidebar.slider('Glucose', 39, 478, 117)
    diabetes = st.sidebar.selectbox('Diabetes', (0, 1))
    heartrate = st.sidebar.slider('Heart Rate', 37, 220, 91)
    cigpday = st.sidebar.slider('Cigarettes Per Day', 0, 90, 20)
    bpmeds = st.sidebar.selectbox('On BP Meds', (0, 1))
    stroke = st.sidebar.selectbox('Stroke', (0, 1))
    hyperten = st.sidebar.selectbox('Hypertension', (0, 1))
    ldlc = st.sidebar.slider('LDL Cholesterol', 0, 200, 100)
    hdlc = st.sidebar.slider('HDL Cholesterol', 0, 100, 50)
    
    return pd.DataFrame({
        'AGE': [age], 'TOTCHOL': [totchol], 'SYSBP': [sysbp],
        'DIABP': [diabp], 'BMI': [bmi], 'CURSMOKE': [cursmoke],
        'GLUCOSE': [glucose], 'DIABETES': [diabetes], 'HEARTRTE': [heartrate],
        'CIGPDAY': [cigpday], 'BPMEDS': [bpmeds], 'STROKE': [stroke],
        'HYPERTEN': [hyperten], 'LDLC': [ldlc], 'HDLC': [hdlc]
    })

input_df = user_input_features()

st.subheader('User Input Parameters')
st.write(input_df)

# Preprocess user input
input_df_scaled = scaler.transform(input_df)

# Convert to tensor
input_tensor = torch.tensor(input_df_scaled, dtype=torch.float32).unsqueeze(1)

# Prediction
with torch.no_grad():
    outputs = transformer_model(input_tensor)
    _, prediction = torch.max(outputs, 1)
    prediction_proba = torch.softmax(outputs, dim=1)

st.subheader('Prediction')
st.write('Risk of Cardiovascular Disease:' if prediction.item() == 1 else 'No Cardiovascular Disease')

st.subheader('Prediction Probability')
st.write(prediction_proba.numpy())

# Feature importance plot (if applicable, otherwise describe key features)
st.subheader('Key Features (based on domain knowledge)')
st.write("""
- AGE: Age of the patient
- TOTCHOL: Total cholesterol level
- SYSBP: Systolic blood pressure
- DIABP: Diastolic blood pressure
- BMI: Body mass index
- CURSMOKE: Current smoker status
- GLUCOSE: Glucose level
- DIABETES: Diabetes status
- HEARTRTE: Heart rate
- CIGPDAY: Cigarettes per day
- BPMEDS: Blood pressure medication status
- STROKE: Stroke history
- HYPERTEN: Hypertension status
- LDLC: LDL cholesterol level
- HDLC: HDL cholesterol level
""")

if __name__ == '__main__':
    st.title("CVD Prediction App with Transformer Model")
