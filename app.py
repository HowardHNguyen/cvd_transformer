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
    heartrate = st.sidebar.slider('Heart Rate', 37, 220, 91)
    cigpday = st.sidebar.slider('Cigarettes Per Day', 0, 90, 20)
    bpmeds = st.sidebar.selectbox('On BP Meds', [0, 1])
    stroke = st.sidebar.selectbox('Stroke', [0, 1])
    hyperten = st.sidebar.selectbox('Hypertension', [0, 1])
    
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
        'HYPERTEN': hyperten
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_data = user_input_features()

st.subheader('User Input Parameters')
st.write(input_data)

# Ensure the correct number of features
required_features = ['AGE', 'TOTCHOL', 'SYSBP', 'DIABP', 'BMI', 'CURSMOKE', 'GLUCOSE', 'DIABETES', 'HEARTRTE', 'CIGPDAY', 'BPMEDS', 'STROKE', 'HYPERTEN']
input_data = input_data[required_features]

# Scale input data
input_data_scaled = scaler.transform(input_data)
input_tensor = torch.tensor(input_data_scaled, dtype=torch.float32)

# Prediction
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
    proba_df = pd.DataFrame(prediction_proba, columns=["No CVD", "CVD"])
    st.write(proba_df)
    
    # Plot prediction probabilities
    fig, ax = plt.subplots()
    ax.bar(proba_df.columns, prediction_proba[0], color=['blue', 'red'])
    ax.set_ylim([0, 1])
    plt.title('Prediction Probability')
    st.pyplot(fig)

    # Feature Importance using Permutation Importance
    st.subheader('Feature Importances (Transformer)')
    
    result = permutation_importance(transformer_estimator, X_train_scaled, y_train, n_repeats=10, random_state=42)
    feature_importance = pd.DataFrame(result.importances_mean, index=input_data.columns, columns=['Importance']).sort_values(by='Importance', ascending=False)
    
    st.write(f"Feature Importances: {feature_importance}")

    fig, ax = plt.subplots()
    feature_importance.plot(kind='bar', ax=ax)
    plt.title('Feature Importances (Permutation Importance)')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    st.pyplot(fig)

    # ROC Curve
    st.subheader('Model Performance (ROC Curve)')
    y_test = joblib.load('transformer_y_test.pkl')  
    y_pred_proba = joblib.load('transformer_y_pred_proba.pkl')
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
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
