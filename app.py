import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc

# Define the TransformerModel class
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
data_url = 'https://github.com/HowardHNguyen/cvd/raw/master/frmgham2.csv'
data = pd.read_csv(data_url)

# Define the input parameters
st.sidebar.header('Please Select Your Parameters')
def user_input_features():
    AGE = st.sidebar.slider('Age', 32, 81, 54)
    TOTCHOL = st.sidebar.slider('Total Cholesterol', 107, 696, 200)
    SYSBP = st.sidebar.slider('Systolic Blood Pressure', 83, 295, 140)
    DIABP = st.sidebar.slider('Diastolic Blood Pressure', 30, 150, 89)
    BMI = st.sidebar.slider('BMI', 14.43, 56.80, 26.77)
    CURSMOKE = st.sidebar.selectbox('Current Smoker', [0, 1])
    GLUCOSE = st.sidebar.slider('Glucose', 39, 478, 117)
    DIABETES = st.sidebar.selectbox('Diabetes', [0, 1])
    HEARTRTE = st.sidebar.slider('Heart Rate', 37, 220, 91)
    CIGPDAY = st.sidebar.slider('Cigarettes Per Day', 0, 90, 20)
    BPMEDS = st.sidebar.selectbox('On BP Meds', [0, 1])
    STROKE = st.sidebar.selectbox('Stroke', [0, 1])
    HYPERTEN = st.sidebar.selectbox('Hypertension', [0, 1])
    LDLC = st.sidebar.slider('LDL Cholesterol', 0, 208, 100)
    HDLC = st.sidebar.slider('HDL Cholesterol', 0, 100, 50)

    features = pd.DataFrame({
        'AGE': [AGE],
        'TOTCHOL': [TOTCHOL],
        'SYSBP': [SYSBP],
        'DIABP': [DIABP],
        'BMI': [BMI],
        'CURSMOKE': [CURSMOKE],
        'GLUCOSE': [GLUCOSE],
        'DIABETES': [DIABETES],
        'HEARTRTE': [HEARTRTE],
        'CIGPDAY': [CIGPDAY],
        'BPMEDS': [BPMEDS],
        'STROKE': [STROKE],
        'HYPERTEN': [HYPERTEN],
        'LDLC': [LDLC],
        'HDLC': [HDLC]
    })
    return features

input_data = user_input_features()

st.subheader('User Input Parameters')
st.write(input_data)

# Preprocess the input data
input_data = scaler.transform(input_data)
input_tensor = torch.tensor(input_data, dtype=torch.float32)

# Prediction button
if st.sidebar.button("PREDICT NOW"):
    with torch.no_grad():
        output = transformer_model(input_tensor.unsqueeze(1))
        _, prediction = torch.max(output, 1)
        prediction_proba = F.softmax(output, dim=1).numpy()

    st.subheader('Prediction')
    st.write('No Cardiovascular Disease' if prediction[0] == 0 else 'Cardiovascular Disease')

    st.subheader('Prediction Probability')
    proba_df = pd.DataFrame(prediction_proba, columns=['No CVD', 'CVD'])
    st.write(proba_df)
    
    fig, ax = plt.subplots()
    ax.bar(proba_df.columns, prediction_proba[0], color=['blue', 'red'])
    plt.ylim([0, 1])
    plt.xlabel('Prediction')
    plt.ylabel('Probability')
    st.pyplot(fig)

    # Model Performance (ROC Curve)
    roc_data = pd.read_csv('roc_data.csv')  # This should contain precomputed ROC data
    fpr = roc_data['fpr']
    tpr = roc_data['tpr']
    roc_auc = roc_data['roc_auc'][0]
    
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'Transformer (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    st.pyplot(plt)
    
    # Feature Importance
    explainer = shap.DeepExplainer(transformer_model, torch.tensor(scaler.transform(data.iloc[:, :-1].values), dtype=torch.float32).unsqueeze(1))
    shap_values = explainer.shap_values(torch.tensor(input_data, dtype=torch.float32).unsqueeze(1))
    
    st.subheader('Feature Importances (Transformer)')
    shap.summary_plot(shap_values, pd.DataFrame(input_data, columns=input_data.columns), plot_type="bar", feature_names=data.columns[:-1], show=False)
    st.pyplot(bbox_inches='tight')
