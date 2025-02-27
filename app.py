import pandas as pd
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import streamlit as st
import numpy as np

#model and dataset 
model_path = 'xgboost_model.json'
dataset_path = 'wine quality_red.csv'

#original dataset to fit the scaler
data = pd.read_csv(dataset_path, sep=";")
features = data.drop("quality", axis=1).values

# Fit the scaler on the original features
scaler = StandardScaler()
scaler.fit(features)

# Load the saved XGBoost model
loaded_model = xgb.XGBClassifier()
loaded_model.load_model(model_path)

#feature names for input labels
feature_names = data.drop("quality", axis=1).columns

# Streamlit app
st.title("Wine Quality Prediction")

#input fields for each feature
input_values = []
for feature in feature_names:
    value = st.number_input(feature, value=0.0)
    input_values.append(value)

# Predict button
if st.button("Predict"):
    # Convert inputs to numpy array and reshape for a single sample
    input_array = np.array(input_values).reshape(1, -1)
    # Scale the input
    input_scaled = scaler.transform(input_array)
    # Make prediction
    prediction = loaded_model.predict(input_scaled)[0]
    proba = loaded_model.predict_proba(input_scaled)[0][1]

    # Convert probability to quality score (0-10 scale)
    quality_score = proba * 10

    # Display the result
    quality = "Good" if prediction == 1 else "Bad"
    st.write(f"Prediction: {quality}, Quality Score: {quality_score:.2f}")

