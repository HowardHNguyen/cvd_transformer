import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Define the Transformer model class
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

# Load the scaler
scaler = joblib.load('scaler.pkl')

# Load the test labels and predicted probabilities
transformer_labels = np.load('transformer_labels.npy')
transformer_probs = np.load('transformer_probs.npy')

# Streamlit app
st.title("Cardiovascular Disease Prediction (Transformer)")

# Define user input fields
st.sidebar.header("Please Select Your Parameters")
age = st.sidebar.slider("Age", 32, 81, 54)
totchol = st.sidebar.slider("Total Cholesterol", 107, 696, 200)
sysbp = st.sidebar.slider("Systolic Blood Pressure", 83, 295, 130)
diabp = st.sidebar.slider("Diastolic Blood Pressure", 30, 150, 82)
bmi = st.sidebar.slider("BMI", 14.43, 56.8, 28.75)
cursmoke = st.sidebar.selectbox("Current Smoker", [0, 1])
glucose = st.sidebar.slider("Glucose", 39, 478, 98)
diabetes = st.sidebar.selectbox("Diabetes", [0, 1])
heartrate = st.sidebar.slider("Heart Rate", 37, 220, 80)
cigpday = st.sidebar.slider("Cigarettes Per Day", 0, 90, 20)
bpm = st.sidebar.selectbox("On BP Meds", [0, 1])
stroke = st.sidebar.selectbox("Stroke", [0, 1])
hyperten = st.sidebar.selectbox("Hypertension", [0, 1])
ldl = st.sidebar.slider("LDL Cholesterol", 0, 200, 100)
hdl = st.sidebar.slider("HDL Cholesterol", 0, 100, 50)

# Collect user inputs into a DataFrame
user_data = pd.DataFrame({
    'AGE': [age],
    'TOTCHOL': [totchol],
    'SYSBP': [sysbp],
    'DIABP': [diabp],
    'BMI': [bmi],
    'CURSMOKE': [cursmoke],
    'GLUCOSE': [glucose],
    'DIABETES': [diabetes],
    'HEARTRTE': [heartrate],
    'CIGPDAY': [cigpday],
    'BPMEDS': [bpm],
    'STROKE': [stroke],
    'HYPERTEN': [hyperten],
    'LDLC': [ldl],
    'HDLC': [hdl]
})

# Scale the user inputs
user_data_scaled = scaler.transform(user_data)

# Convert user inputs to tensor
input_tensor = torch.tensor(user_data_scaled, dtype=torch.float32)

# Prediction button
if st.sidebar.button("PREDICT NOW"):
    with torch.no_grad():
        output = transformer_model(input_tensor.unsqueeze(1))
        _, prediction = torch.max(output, 1)
        prediction_proba = F.softmax(output, dim=1).numpy()

    st.write("## Prediction")
    if prediction.item() == 0:
        st.write("### No Cardiovascular Disease")
    else:
        st.write("### Cardiovascular Disease")

    st.write("### Prediction Probability")
    st.write(pd.DataFrame(prediction_proba, columns=["No CVD", "CVD"]))

    # Plot prediction probability
    st.write("### Prediction Probability")
    fig, ax = plt.subplots()
    ax.bar(["No CVD", "CVD"], prediction_proba[0], color=["blue", "red"])
    ax.set_ylim([0, 1])
    ax.set_ylabel("Probability")
    st.pyplot(fig)

    # Plot ROC Curve
    st.write("### Model Performance (ROC Curve)")
    fpr, tpr, _ = roc_curve(transformer_labels, transformer_probs)
    roc_auc = auc(fpr, tpr)
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

    # Feature importances
    st.write("### Feature Importances")
    importances = transformer_model.fc1.weight.detach().cpu().numpy().flatten()
    feature_names = user_data.columns
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    fig, ax = plt.subplots()
    ax.barh(importance_df['Feature'], importance_df['Importance'], color='blue')
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    ax.set_title('Feature Importances (Transformer)')
    st.pyplot(fig)
