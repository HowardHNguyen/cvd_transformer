import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import shap

# Define the TransformerModel class
class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, 128)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 1, 128))
        encoder_layers = nn.TransformerEncoderLayer(d_model=128, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=3)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x + self.pos_encoder
        x = self.transformer_encoder(x.unsqueeze(1)).squeeze(1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# Load the model
transformer_model = TransformerModel(input_dim=13, num_classes=2)
transformer_model.load_state_dict(torch.load('transformer_model.pth', map_location=torch.device('cpu')))
transformer_model.eval()

# Load the scaler
scaler = joblib.load('scaler.pkl')

# Your Streamlit app code continues...

# Function to make predictions and display results
def predict_and_display(input_data):
    input_data_scaled = scaler.transform(input_data)
    input_tensor = torch.tensor(input_data_scaled, dtype=torch.float32)
    
    # Prediction
    with torch.no_grad():
        prediction = transformer_model(input_tensor).numpy()
    
    # Display prediction results
    st.write(f"Prediction: {prediction}")

# Streamlit UI components
st.title("Cardiovascular Disease Prediction (Transformer)")
age = st.slider("Age", 32, 81, 54)
totchol = st.slider("Total Cholesterol", 107, 696, 200)
sysbp = st.slider("Systolic Blood Pressure", 83, 295, 140)
diabp = st.slider("Diastolic Blood Pressure", 38, 150, 89)
bmi = st.slider("BMI", 14.43, 56.80, 26.77)
cursmoke = st.selectbox("Current Smoker", [0, 1])
glucose = st.slider("Glucose", 39, 478, 117)
diabetes = st.selectbox("Diabetes", [0, 1])
heartrate = st.slider("Heart Rate", 37, 220, 91)
cigpday = st.slider("Cigarettes Per Day", 0, 90, 20)
bpmeds = st.selectbox("On BP Meds", [0, 1])
stroke = st.selectbox("Stroke", [0, 1])
hyperten = st.selectbox("Hypertension", [0, 1])

# Collect input data
input_data = np.array([[age, totchol, sysbp, diabp, bmi, cursmoke, glucose, diabetes, heartrate, cigpday, bpmeds, stroke, hyperten]])

# Predict and display results
if st.button("Predict"):
    predict_and_display(input_data)

# Feature Importances and ROC Curve (assuming you have precomputed these)
# Add your existing logic here for displaying feature importances and ROC curve


# Feature Importances
st.subheader('Feature Importances (Transformer)')
try:
    result = permutation_importance(transformer_model, X_train_scaled, y_train, n_repeats=10, random_state=42, scoring='accuracy')
    importances = pd.Series(result.importances_mean, index=input_data.columns)
    st.write(importances)
    fig, ax = plt.subplots()
    importances.plot.bar(ax=ax)
    ax.set_title('Feature Importances (Permutation Importance)')
    ax.set_ylabel('Importance')
    st.pyplot(fig)
except Exception as e:
    st.error(f"Error calculating feature importances: {e}")

# ROC Curve
st.subheader('Model Performance (ROC Curve)')
try:
    y_test = joblib.load('y_test.pkl')
    y_pred_proba = joblib.load('transformer_y_pred_proba.pkl')
    
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='blue', lw=2, label='Transformer (AUC = %0.4f)' % roc_auc)
    ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc='lower right')
    st.pyplot(fig)
except Exception as e:
    st.error(f"Error loading or plotting ROC curve data: {e}")
