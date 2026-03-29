import pandas as pd
import streamlit as st

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline

from src.data_loader import load_data, save_data
from src.model import split_train_test, log_transform_func, inverse_log_transform_func, make_log_transformer, make_preprocessor, make_pipeline, make_model, save_model, load_model
from src.evaluate import evaluate_model, plot_predictions, plot_feature_importance

import joblib

# Setting up header
st.set_page_config(page_title='House Price Predictor', layout='centered')
st.title("🏠 House Price Prediction App")
st.markdown("Enter your house details to get an estimated price for your house.")

# Loading trained model
@st.cache_resource
def load_model():
    return joblib.load('notebooks/model/model.pkl')

model = load_model()

# Getting input from user
grade_of_the_house = st.slider("Grade of the House", min_value=1, max_value=15, value=3)
living_area = st.slider("Living Area (sqft)", min_value=200.0, max_value=15000.0, value=1800.0, step=10.0)
living_area_renov = st.slider("Living Area after Renovation (sqft)", min_value=200.0, max_value=15000.0, value=1800.0, step=10.0)
bathrooms = st.slider("Number of Bathrooms", min_value=1, max_value=10, value=3, step=1)
area_excluding_basement = st.slider("Area of the House (excluding basement)", min_value=200.0, max_value=15000.0, value=1800.0, step=10.0)
lat = st.number_input(
    label="Enter Latitude",
    min_value=-90.0,
    max_value=90.0,
    value=31.3723,
    format="%.4f"
)
floors = st.slider("Number of Floors", min_value=1, max_value=5, value=1)
bedrooms = st.slider("Number of Bedrooms", min_value=0, max_value=50, value=3, step=1)
views = st.slider("Number of Views", min_value=0, max_value=5, value=2, step=1)

# Creating input for model
input_dict = pd.DataFrame({
    'grade of the house': [grade_of_the_house],
    'living area': [living_area],
    'living_area_renov': [living_area_renov],
    'number of bathrooms': [bathrooms],
    'Area of the house(excluding basement)': [area_excluding_basement],
    'Lattitude': [lat],
    'number of floors': [floors],
    'number of bedrooms': [bedrooms],
    'number of views': [views],
})

# Making Prediction
if st.button("Predict"):
    prediction_log = model.predict(input_dict)
    prediction = model.inverse_func(prediction_log)
    st.success(f"Estimated House Price: ${prediction[0]:.2f}")