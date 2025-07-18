from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from joblib import dump, load

def train_random_forest(X_train, y_train, n_estimators=1000, random_state=15):
    """
    Train a Random Forest Regressor.
    """
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model

def train_decision_tree(X_train, y_train, random_state=15, max_depth=21, min_samples_split=14):
    """
    Train a Decision Tree Regressor.
    """
    model = DecisionTreeRegressor(random_state=random_state, max_depth=max_depth, min_samples_split=min_samples_split)
    model.fit(X_train, y_train)
    return model

def save_model(model, filepath):
    """
    Save the trained model to a file.
    """
    dump(model, filepath)

def load_model(filepath):
    """
    Load a trained model from a file.
    """
    return load(filepath)
