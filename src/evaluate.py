import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import pandas as pd

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model and return MAE and predictions.
    """
    y_pred = model.predict(X_test)
    
    y_test_in = model.inverse_func(y_test)
    y_pred_in = model.inverse_func(y_pred)
    
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_sqauared_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    
    return y_test_in, y_pred_in, r2, mae, mse, mape

def plot_predictions(y_test, y_pred, save_path=None):
    """
    Plot actual vs predicted values.
    """
    sns.scatterplot(x=y_test, y=y_pred)
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title("Actual vs Predicted Prices")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.savefig(save_path)
    plt.show()
    plt.close()

def plot_feature_importance(model, save_path=None):
    """
    Plot feature importances of the model.
    """
    importances = model.regressor_['regressor'].feature_importances_
    feat_importances = pd.Series(importances, index=features)
    
    plt.figure(figsize=(10, 6))
    feat_importances.nlargest(10).plot(kind='barh', color='skyblue')
    plt.title('Top 10 Most Important Features')
    plt.xlabel('Relative Importance Score')
    plt.gca().invert_yaxis() # Highest importance at the top
    plt.savefig(save_path)
    plt.show()
