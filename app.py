import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import joblib
from sklearn.metrics import roc_curve, auc

# Define the model
class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_classes, d_model=128, max_seq_length=1, nhead=8, num_layers=4):
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

# Load the evaluation metrics
transformer_labels = np.load('transformer_labels.npy')
transformer_probs = np.load('transformer_probs.npy')

# Streamlit UI
st.title("Cardiovascular Disease Prediction (Transformer)")

# User inputs
age = st.sidebar.slider("Age", 32, 81, 54)
totchol = st.sidebar.slider("Total Cholesterol", 107, 696, 200)
sysbp = st.sidebar.slider("Systolic Blood Pressure", 83, 295, 140)
diabp = st.sidebar.slider("Diastolic Blood Pressure", 30, 150, 89)
bmi = st.sidebar.slider("BMI", 14.43, 56.8, 26.77)
cursmoke = st.sidebar.selectbox("Current Smoker", [0, 1])
glucose = st.sidebar.slider("Glucose", 39, 478, 117)
diabetes = st.sidebar.selectbox("Diabetes", [0, 1])
heartrate = st.sidebar.slider("Heart Rate", 37, 220, 91)
cigpday = st.sidebar.slider("Cigarettes Per Day", 0, 90, 20)
bpmeds = st.sidebar.selectbox("On BP Meds", [0, 1])
stroke = st.sidebar.selectbox("Stroke", [0, 1])
hyperten = st.sidebar.selectbox("Hypertension", [0, 1])
ldlc = st.sidebar.slider("LDL Cholesterol", 0, 208, 100)
hdlc = st.sidebar.slider("HDL Cholesterol", 0, 100, 50)

# Create input dataframe
input_data = pd.DataFrame({
    'AGE': [age], 'TOTCHOL': [totchol], 'SYSBP': [sysbp], 'DIABP': [diabp], 'BMI': [bmi],
    'CURSMOKE': [cursmoke], 'GLUCOSE': [glucose], 'DIABETES': [diabetes], 'HEARTRTE': [heartrate],
    'CIGPDAY': [cigpday], 'BPMEDS': [bpmeds], 'STROKE': [stroke], 'HYPERTEN': [hyperten],
    'LDLC': [ldlc], 'HDLC': [hdlc]
})

# Scale the input
input_scaled = scaler.transform(input_data)

# Convert to tensor
input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

# Prediction button
if st.sidebar.button("PREDICT NOW"):
    with torch.no_grad():
        output = transformer_model(input_tensor.unsqueeze(1))
        _, prediction = torch.max(output, 1)
        prediction_proba = F.softmax(output, dim=1).numpy()

    st.subheader("Prediction")
    if prediction.item() == 1:
        st.write("Cardiovascular Disease")
    else:
        st.write("No Cardiovascular Disease")

    st.subheader("Prediction Probability")
    prob_df = pd.DataFrame(prediction_proba, columns=["No CVD", "CVD"])
    st.write(prob_df)
    st.bar_chart(prob_df.T)

# ROC Curve
st.subheader("Model Performance (ROC Curve)")
fpr, tpr, _ = roc_curve(transformer_labels, transformer_probs)
roc_auc = auc(fpr, tpr)
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

# Feature Importances
st.subheader("Feature Importances (Transformer)")

# Assuming the transformer_model has a linear layer named 'embedding'
importances = np.abs(transformer_model.embedding.weight.detach().numpy().flatten())
feature_importances = pd.DataFrame({'feature': input_data.columns, 'importance': importances})

# Sort the feature importances
feature_importances = feature_importances.sort_values(by='importance', ascending=False)

# Plot the feature importances
fig, ax = plt.subplots()
ax.barh(feature_importances['feature'], feature_importances['importance'])
ax.set_xlabel('Importance')
ax.set_ylabel('Feature')
ax.set_title('Risk Factors / Feature Importances (Transformer)')
st.pyplot(fig)
