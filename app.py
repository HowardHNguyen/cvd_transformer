import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.inspection import permutation_importance
import joblib
from sklearn.base import BaseEstimator, ClassifierMixin

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

class TransformerEstimator(BaseEstimator, ClassifierMixin):
    def __init__(self, model):
        self.model = model
    
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self
    
    def predict(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predictions = torch.max(outputs, 1)
        return predictions.numpy()
    
    def predict_proba(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = F.softmax(outputs, dim=1)
        return probabilities.numpy()

# Load the model and data
transformer_model = TransformerModel(input_dim=13, num_classes=2)
try:
    transformer_model.load_state_dict(torch.load('transformer_model.pth', map_location=torch.device('cpu')))
    st.success("Model loaded successfully")
except RuntimeError as e:
    st.error(f"Error loading model state_dict: {e}")
transformer_model.eval()

# Load the scaler
scaler = pd.read_pickle('scaler.pkl')

# Load the training data
X_train_scaled = joblib.load('X_train_scaled.pkl')
y_train = joblib.load('y_train.pkl')

# Wrap the transformer model
transformer_estimator = TransformerEstimator(transformer_model)
transformer_estimator.fit(X_train_scaled, y_train)

# Streamlit app
st.title("Cardiovascular Disease Prediction (Transformer)")

# User input features
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
    heartrte = st.sidebar.slider('Heart Rate', 37, 220, 91)
    cigpday = st.sidebar.slider('Cigarettes Per Day', 0, 90, 20)
    bpmeds = st.sidebar.selectbox('On BP Meds', [0, 1])
    stroke = st.sidebar.selectbox('Stroke', [0, 1])
    hyp = st.sidebar.selectbox('Hypertension', [0, 1])
    
    data = {'AGE': age,
            'TOTCHOL': totchol,
            'SYSBP': sysbp,
            'DIABP': diabp,
            'BMI': bmi,
            'CURSMOKE': cursmoke,
            'GLUCOSE': glucose,
            'DIABETES': diabetes,
            'HEARTRTE': heartrte,
            'CIGPDAY': cigpday,
            'BPMEDS': bpmeds,
            'STROKE': stroke,
            'HYPERTEN': hyp}
    
    features = pd.DataFrame(data, index=[0])
    return features

input_data = user_input_features()

# Display user input features
st.subheader('User Input Parameters')
st.write(input_data)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Prediction
prediction = transformer_estimator.predict(input_data_scaled)
prediction_proba = transformer_estimator.predict_proba(input_data_scaled)

# Display prediction
st.subheader('Prediction')
cvd_labels = np.array(['No Cardiovascular Disease', 'Cardiovascular Disease'])
st.write(cvd_labels[prediction])

# Display prediction probability
st.subheader('Prediction Probability')
st.write(prediction_proba)

fig, ax = plt.subplots()
ax.bar(cvd_labels, prediction_proba[0], color=['blue', 'red'])
plt.xlabel('CVD Status')
plt.ylabel('Probability')
plt.title('Prediction Probability')
st.pyplot(fig)

# Feature Importances
st.subheader('Feature Importances (Transformer)')
try:
    result = permutation_importance(transformer_estimator, X_train_scaled, y_train, n_repeats=10, random_state=42, scoring='accuracy')
    importances = pd.Series(result.importances_mean, index=X_train.columns)
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
