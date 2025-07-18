# 🏡 Real Estate Price Prediction App

This project is a **machine learning-based web application** that predicts house prices based on user input features. It uses a **Random Forest Regressor** trained on housing data and is deployed with **Streamlit** for interactive predictions.

---

## 📌 Features

- Predict house prices using features like:
  - Number of bedrooms and bathrooms
  - Living and lot area
  - Floors, view, condition, and grade
  - Construction year and renovation year
  - Location (City)
- User-friendly web interface with sliders and text inputs
- Feature importance visualization
- MAE evaluation on test set

---


---

## ⚙️ How to Run the App

### 1. Clone the Repository

```bash
git clone https://github.com/simrxn56/House%20Price%20Prediction.git
cd "House Price Prediction"
```

### 2. Install Requirements 

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit App

```bash
streamlit run app/app.py
```

---

## 💡 Example Prediction

You can input data such as:
- Bedrooms: 3
* Bathrooms: 2
+ Living Area: 1500 sqft
- Grade: 7
* City: Seattle

And the model will output an estimated price like:
> And the model will output an estimated price like:
> `Predicted Price: $530,000`

---

## 📦 Requirements
* Python 3.8+
- pandas
+ numpy
- scikit-learn
* matplotlib
+ seaborn
- streamlit
* joblib

###Install using: 
```bash
pip install -r requirements.txt
```

---

## 📊 Model Info

- Algorithm: Random Forest Regressor
* Evaluation Metric: Mean Absolute Error (MAE)
+ No feature scaling required due to tree-based model
- City column one-hot encoded for better model performance

---

## ✅ Future Improvements

- Use XGBoost or LightGBM for better accuracy
+ Deploy using Streamlit Cloud or Hugging Face Spaces
- Add model versioning and logging

---

## 📬 Contact
If you found this useful or have questions, feel free to reach out!
- 📧 mr.ramgharia2054@gmail.com
* 🔗 LinkedIn [Simranjit Singh](https://www.linkedin.com/in/simranjit-singh-91674a358/)
