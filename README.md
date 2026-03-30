# 🏠 House Price Prediction App

A machine learning web application that predicts residential property prices from user inputs. Built with a fully modular sklearn pipeline — EDA, preprocessing, log-transformed target regression, evaluation, and the Streamlit UI are all cleanly separated into dedicated modules.

---

## Model Performance

| Metric | Score |
|--------|-------|
| R² Score | **0.8121** |
| MAE | **$72,632.76** |
| MAPE | **14.13%** |
| MSE | 13,592,384,279 |

The model accounts for **81% of total price variance**. On average, predictions deviate by ~$72K from actual prices — strong performance relative to the extreme scale of the target variable (house prices range from affordable to luxury).

---

## How It Works

### EDA (`notebooks/01_eda.ipynb`)
- Identified right-skewed columns: `Price`, `living area`, `living_area_renov`, `Area of the house(excluding basement)`, `lot area`, `lot_area_renov`, `Area of the basement`
- Applied `log1p` transformation to skewed columns and removed outliers using the IQR method
- Analyzed both **Pearson** (linear) and **Spearman** (monotonic) correlations to handle multicollinearity
- Selected **RandomForestRegressor** specifically because it handles correlated features well — different feature subsets are chosen at each split, preventing any single correlated feature from dominating

### Modelling (`notebooks/02_modelling.ipynb`)
Full sklearn pipeline:
```
User Input
    ↓
ColumnTransformer — log1p applied to: living area, living_area_renov, Area excl. basement
    ↓
RandomForestRegressor (n_estimators=100, max_depth=30)
    ↓
TransformedTargetRegressor — log1p on target during training, expm1 on predictions
    ↓
Predicted Price (original dollar scale)
```

The `TransformedTargetRegressor` wrapper automates log transformation of the target during training and inverse transformation at prediction time — no manual steps needed in the app.

---

## Input Features

| Feature | Range |
|---------|-------|
| Grade of the House | 1 – 15 |
| Living Area (sqft) | 200 – 15,000 |
| Living Area after Renovation (sqft) | 200 – 15,000 |
| Number of Bathrooms | 1 – 10 |
| Area excluding Basement (sqft) | 200 – 15,000 |
| Latitude | -90.0 – 90.0 |
| Number of Floors | 1 – 5 |
| Number of Bedrooms | 0 – 50 |
| Number of Views | 0 – 5 |

Features log-transformed inside the pipeline: `living area`, `living_area_renov`, `Area of the house(excluding basement)`

---

## Project Structure

```
House-Price-Prediction/
│
├── app/
│   └── app.py              # Streamlit UI — inputs, model loading, prediction
│
├── src/
│   ├── data_loader.py      # load_data() and save_data() utilities
│   ├── model.py            # Pipeline builder: log transformer → ColumnTransformer
│   │                       # → RandomForest → TransformedTargetRegressor
│   └── evaluate.py         # R², MAE, MSE, MAPE + prediction and feature importance plots
│
├── main/
│   └── main.py             # End-to-end training script: load → preprocess → train → evaluate → save
│
├── notebooks/
│   ├── 01_eda.ipynb        # Distributions, skewness, IQR outlier removal, correlation analysis
│   ├── 02_modelling.ipynb  # Pipeline construction, training, evaluation, feature importance
│   └── model/
│       └── model.pkl       # ⚠️ Too large for GitHub — download below
│
├── data/
│   ├── data.csv            # Raw dataset
│   └── clean_data.csv      # After log transform + IQR outlier removal
│
├── outputs/                # Saved plots from notebooks
├── requirements.txt
└── LICENSE
```

---

## ⬇️ Download the Trained Model

The model file (`model.pkl`) is too large to store on GitHub.

**[Download model.pkl from Google Drive](https://drive.google.com/file/d/1eOjuVLTzqX5fsBDNOsKTHxxQ4V8uLC3q/view?usp=drive_link)**

After downloading, place it at:
```
notebooks/model/model.pkl
```

---

## Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/simrxn56/House-Price-Prediction.git
cd House-Price-Prediction
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Download the model**

Download `model.pkl` from the link above and place it at `notebooks/model/model.pkl`

**4. Run the app**
```bash
streamlit run app/app.py
```

**5. (Optional) Retrain the model yourself**
```bash
python main/main.py
```

---

## Example Prediction

| Input | Value |
|-------|-------|
| Grade | 7 |
| Living Area | 1,500 sqft |
| Bathrooms | 2 |
| Bedrooms | 3 |
| Floors | 1 |
| Latitude | 47.5112 |

**→ Estimated Price: ~$530,000**

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Pipeline-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)

- **ML:** scikit-learn — RandomForestRegressor, Pipeline, ColumnTransformer, TransformedTargetRegressor
- **Data:** pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Deployment:** Streamlit, joblib

---

## Author

**Simranjit Singh**
[GitHub](https://github.com/simrxn56) · [LinkedIn](https://www.linkedin.com/in/simranjit-singh-s2054/) · [Kaggle](https://www.kaggle.com/simranjit205456)
