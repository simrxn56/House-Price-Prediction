import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline

from src.data_loader import load_data, save_data
from src.model import split_train_test, log_transform_func, inverse_log_transform_func, make_log_transformer, make_preprocessor, make_pipeline, make_model, save_model, load_model
from src.evaluate import evaluate_model, plot_predictions, plot_feature_importance

from joblib import dump, load

import os

# ----- Step 1: Load Data -----
data_path = "../data/data.csv"
df = load_data(data_path)

features = ['grade of the house',
            'living area',
             'living_area_renov',
             'number of bathrooms',
             'Area of the house(excluding basement)',
             'Lattitude',
             'number of floors',
             'number of bedrooms',
             'number of views'
           ]
features_to_transform = ['living_area_renov', 'Area of the house(excluding basement)', 'living area'] 

# ----- Step 2: Preprocessing -----
log_transformer = make_log_transformer(log_transform_func, inverse_log_transform_func)

preprocessor = make_preprocessor(log_transformer, features_to_transform)

pipeline = make_pipeline(preprocessor)

X = df[features]
y = np.log1p(df['Price'])
X_train, X_test, y_train, y_test = split_train_test(X, y)

# ----- Step 3: Train Model -----
model = make_model(pipeline)
model.fit(X_train, y_train)

# ----- Step 4: Save Model -----
model_dir = "../notebooks/model"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "model.pkl")
save_model(model, model_path)

# ----- Step 5: Evaluate Model -----
y_test_in, y_pred_in, r2, mae, mse, mape = evaluate_model(model, X_test, y_test)
print(f"R2 Score: {r2:.4f}")
print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"MAPE: {mape:.4f}")

# ----- Step 6: Plot Results -----
plot_predictions(y_test_in, y_pred_in, save_path="../outputs/notebook 4/actual vs predicted prices.png")
plot_feature_importance(model, feature_names=X.columns, save_path="plots/feature_importance.png")
