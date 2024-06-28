import torch
import torch.nn as nn

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

# Instantiate the model with the correct dimensions
transformer_model = TransformerModel(input_dim=13, num_classes=2)

# Load the state dict
state_dict = torch.load('transformer_model.pth', map_location=torch.device('cpu'))

# Print the keys in the state dictionary
print("State dict keys:")
for key in state_dict.keys():
    print(key, state_dict[key].shape)

# Print the keys in the model's state dictionary
print("\nModel state dict keys:")
for key, param in transformer_model.state_dict().items():
    print(key, param.shape)

# Attempt to load the state dict into the model and catch the error if it fails
try:
    missing_keys, unexpected_keys = transformer_model.load_state_dict(state_dict, strict=False)
    print(f"Missing keys: {missing_keys}")
    print(f"Unexpected keys: {unexpected_keys}")
except RuntimeError as e:
    print(str(e))

# Continue with Streamlit app if loading is successful
import streamlit as st
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

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
