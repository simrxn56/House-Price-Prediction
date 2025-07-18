from src.data_loader import load_data, save_data
from src.preprocessing import encode_categoricals, split_features_target, split_train_test
from src.model import train_random_forest, save_model
from src.evaluate import evaluate_model, plot_predictions, plot_feature_importance

import os

# ----- Step 1: Load Data -----
data_path = "data/processed/housing_clean.csv"
df = load_data(data_path)

# ----- Step 2: Preprocessing -----
df_encoded = encode_categoricals(df)
X, y = split_features_target(df_encoded, target_column="price")
X_train, X_test, y_train, y_test = split_train_test(X, y)

# ----- Step 3: Train Model -----
model = train_random_forest(X_train, y_train)

# ----- Step 4: Save Model -----
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "random_forest_model.joblib")
save_model(model, model_path)

# ----- Step 5: Evaluate Model -----
mae, predictions = evaluate_model(model, X_test, y_test)
print(f"MAE on test data: {mae:.2f}")

# ----- Step 6: Plot Results -----
plot_predictions(y_test, predictions, save_path="plots/predictions.png")
plot_feature_importance(model, feature_names=X.columns, save_path="plots/feature_importance.png")
