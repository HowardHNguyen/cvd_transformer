import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.inspection import permutation_importance

# Define the TransformerModel class
class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, 128)
        self.transformer = nn.Transformer(nhead=8, num_encoder_layers=3)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(0)  # Add batch dimension
        x = self.transformer(x)
        x = x.squeeze(0)  # Remove batch dimension
        x = self.fc(x)
        return x

# Load the model and scaler
transformer_model = TransformerModel(input_dim=13, num_classes=2)
transformer_model.load_state_dict(torch.load('transformer_model.pth', map_location=torch.device('cpu')))
transformer_model.eval()

scaler = joblib.load('scaler.pkl')

st.title('Cardiovascular Disease Prediction (Transformer)')

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

try:
    input_data_scaled = scaler.transform(input_data)
except Exception as e:
    st.error(f"Error in scaling input data: {e}")
    st.stop()

# Prediction
prediction = transformer_model(torch.tensor(input_data_scaled, dtype=torch.float32)).argmax(dim=1).item()
prediction_proba = transformer_model(torch.tensor(input_data_scaled, dtype=torch.float32)).softmax(dim=1).detach().numpy()

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
