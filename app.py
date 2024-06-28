import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Define the TransformerModel class
class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, 128)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 128))
        encoder_layers = nn.TransformerEncoderLayer(d_model=128, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=3)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x + self.pos_encoder
        x = self.transformer_encoder(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# Instantiate the model with the correct dimensions
transformer_model = TransformerModel(input_dim=13, num_classes=2)

# Load the state dict
state_dict = torch.load('transformer_model.pth', map_location=torch.device('cpu'))

# Load the state dict into the model
missing_keys, unexpected_keys = transformer_model.load_state_dict(state_dict, strict=False)
print(f'Missing keys: {missing_keys}')
print(f'Unexpected keys: {unexpected_keys}')

# Load the scaler
scaler = joblib.load('scaler.pkl')

# Function to make predictions and display results
def predict_and_display(input_data):
    input_data_scaled = scaler.transform(input_data)
    input_tensor = torch.tensor(input_data_scaled, dtype=torch.float32)
    
    # Prediction
    with torch.no_grad():
        prediction = transformer_model(input_tensor).numpy()
    
    # Calculate probabilities
    probabilities = nn.Softmax(dim=1)(torch.tensor(prediction)).numpy()

    # Display prediction results
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"Prediction: {'CVD' if np.argmax(probabilities) == 1 else 'No CVD'}")
        st.write(f"Prediction Probability: No CVD: {probabilities[0][0]:.4f}, CVD: {probabilities[0][1]:.4f}")

    # Feature Importances
    feature_importances = np.random.rand(13)  # Replace with actual feature importances calculation
    feature_names = ['AGE', 'TOTCHOL', 'SYSBP', 'DIABP', 'BMI', 'CURSMOKE', 'GLUCOSE', 'DIABETES', 'HEARTRTE', 'CIGPDAY', 'BPMEDS', 'STROKE', 'HYPERTEN']
    importance_dict = {name: importance for name, importance in zip(feature_names, feature_importances)}

    with col2:
        st.write("Feature Importances:")
        sorted_importances = sorted(importance_dict.items(), key=lambda item: item[1], reverse=True)
        for feature, importance in sorted_importances:
            st.write(f"{feature}: {importance:.4f}")

    # Plot Feature Importances
    fig, ax = plt.subplots()
    y_pos = np.arange(len(feature_names))
    ax.barh(y_pos, feature_importances, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_names)
    ax.invert_yaxis()
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importances')

    with col2:
        st.pyplot(fig)

    # ROC Curve
    y_true = np.random.randint(2, size=100)  # Replace with actual y_true
    y_scores = np.random.rand(100, 2)[:, 1]  # Replace with actual y_scores from model
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = roc_auc_score(y_true, y_scores)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc="lower right")

    st.pyplot(fig)

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
