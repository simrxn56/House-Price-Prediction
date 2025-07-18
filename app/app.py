import pandas as pd
import streamlit as st
import joblib

# Setting up header
st.set_page_config(page_title='House Price Predictor', layout='centered')
st.title("🏠 House Price Prediction App")
st.markdown("Enter your house details to get an estimated price for your house.")

# Loading trained model
@st.cache_resource
def load_model():
    return joblib.load('C:/Users/LENOVO/OneDrive/Desktop/PYTHON/House Price Prediction/notebooks/model/random_forest_model.pkl')

model = load_model()

# Getting input from user
bedrooms = st.slider("Bedrooms", min_value=0, max_value=10, value=3)
bathrooms = st.slider("Bathrooms", min_value=0.0, max_value=10.0, step=0.25, value=2.0)
sqft_living = st.slider("Living Area (sqft)", min_value=500, max_value=5000, value=1800, step=100)
sqft_lot = st.slider("Lot Size (sqft)", min_value=500, max_value=15000, value=4000, step=100)
floors = st.slider("Floors", min_value=0.0, max_value=4.0, value=1.0, step=0.5)
waterfront = st.selectbox("Waterfront", [0, 1])
view = st.slider("View", min_value=0, max_value=4, value=1)
condition = st.slider("Condition", min_value=0, max_value=5, value=3)
sqft_above = st.slider("Above Ground Area (sqft)", min_value=500, max_value=5000, value=2000, step=100)
sqft_basement = st.slider("Basement Area (sqft)", min_value=0, max_value=3000, value=500)
yr_built = st.slider("Year Built", min_value=1900, max_value=2025, value=1990)
yr_renovated = st.slider("Year Renovated", min_value=1900, max_value=2025, value=0)

cities = ['Algona', 'Auburn', 'Beaux Arts Village', 'Bellevue', 'Black Diamond',
          'Bothell', 'Burien', 'Carnation', 'Clyde Hill', 'Covington', 'Des Moines',
          'Duvall', 'Enumclaw', 'Fall City', 'Federal Way', 'Inglewood-Finn Hill',
          'Issaquah', 'Kenmore', 'Kent', 'Kirkland', 'Lake Forest Park',
          'Maple Valley', 'Medina', 'Mercer Island', 'Milton', 'Newcastle',
          'Normandy Park', 'North Bend', 'Pacific', 'Preston', 'Ravensdale',
          'Redmond', 'Renton', 'Sammamish', 'SeaTac', 'Seattle', 'Shoreline',
          'Skykomish', 'Snoqualmie', 'Snoqualmie Pass', 'Tukwila', 'Vashon',
          'Woodinville', 'Yarrow Point']

city = st.selectbox("City", cities)

# Creating input for model
input_dict = pd.DataFrame({
    'bedrooms': [bedrooms],
    'bathrooms': [bathrooms],
    'sqft_living': [sqft_living],
    'sqft_lot': [sqft_lot],
    'floors': [floors],
    'waterfront': [waterfront],
    'view': [view],
    'condition': [condition],
    'sqft_above': [sqft_above],
    'sqft_basement': [sqft_basement],
    'yr_built': [yr_built],
    'yr_renovated': [yr_renovated],
})

for c in cities:
    col = f'city_{c}'
    input_dict[col] = [1 if c == city else 0]

# Making Prediction
if st.button("Predict"):
    prediction = model.predict(input_dict)[0]
    st.success(f"Estimated House Price: ${prediction:,.2f}")