import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import pandas as pd

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model and return MAE and predictions.
    """
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    return mae, predictions

def plot_predictions(y_test, predictions, save_path=None):
    """
    Plot actual vs predicted values.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, predictions, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Actual vs Predicted Prices")
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_feature_importance(model, feature_names, save_path=None):
    """
    Plot feature importances of the model.
    """
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'], importance_df['Importance'])
    plt.xlabel("Importance")
    plt.title("Feature Importances")
    plt.gca().invert_yaxis()
    if save_path:
        plt.savefig(save_path)
    plt.show()
