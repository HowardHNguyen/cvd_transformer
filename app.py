import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
import shap

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

# Load the model and data
transformer_model = TransformerModel(input_dim=15, num_classes=2)
transformer_model.load_state_dict(torch.load('transformer_model.pth', map_location=torch.device('cpu')))
transformer_model.eval()

scaler = pd.read_pickle('scaler.pkl')

# Streamlit app
st.title('Cardiovascular Disease Prediction (Transformer) by Howard Nguyen')

# Input parameters
st.sidebar.header('Please Select Your Parameters')
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
hypertension = st.sidebar.selectbox('Hypertension', [0, 1])
ldlc = st.sidebar.slider('LDL Cholesterol', 0, 208, 100)
hdlc = st.sidebar.slider('HDL Cholesterol', 0, 100, 50)

# Create dataframe for input features
input_data = pd.DataFrame({
    'AGE': [age], 'TOTCHOL': [totchol], 'SYSBP': [sysbp], 'DIABP': [diabp],
    'BMI': [bmi], 'CURSMOKE': [cursmoke], 'GLUCOSE': [glucose], 'DIABETES': [diabetes],
    'HEARTRTE': [heartrate], 'CIGPDAY': [cigpday], 'BPMEDS': [bpmeds], 'STROKE': [stroke],
    'HYPERTEN': [hypertension], 'LDLC': [ldlc], 'HDLC': [hdlc]
})

st.write('## User Input Parameters')
st.write(input_data)

# Prediction button
if st.sidebar.button("PREDICT NOW"):
    with torch.no_grad():
        input_data = scaler.transform(input_data)
        input_tensor = torch.tensor(input_data, dtype=torch.float32)
        output = transformer_model(input_tensor.unsqueeze(1))
        _, prediction = torch.max(output, 1)
        prediction_proba = F.softmax(output, dim=1).numpy()

        st.write('## Prediction')
        if prediction.item() == 0:
            st.write('No Cardiovascular Disease')
        else:
            st.write('Cardiovascular Disease')

        st.write('## Prediction Probability')
        proba_df = pd.DataFrame(prediction_proba, columns=['No CVD', 'CVD'])
        st.write(proba_df)
        
        fig, ax = plt.subplots()
        proba_df.plot(kind='bar', ax=ax)
        st.pyplot(fig)

        st.write('## Model Performance (ROC Curve)')
        roc_data = pd.read_pickle('roc_data.pkl')
        fpr, tpr, roc_auc = roc_data['fpr'], roc_data['tpr'], roc_data['roc_auc']
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, color='blue', lw=2, label=f'Transformer (AUC = {roc_auc:.4f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.legend(loc="lower right")
        st.pyplot(fig)

        st.write('## Feature Importances (Transformer)')
        explainer = shap.Explainer(transformer_model, input_tensor)
        shap_values = explainer(input_tensor)
        shap.summary_plot(shap_values, input_tensor, feature_names=input_data.columns, plot_type="bar", show=False)
        st.pyplot(bbox_inches='tight')
