import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from joblib import dump, load

def split_train_test(X, y, test_size=0.25, random_state=42):
    """
    Split features and target into training and testing sets.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def log_transform_func(X):
    # Use np.log1p for numerical stability (handles zero values)
    return np.log1p(X)

# You might also want an inverse function for regression targets
def inverse_log_transform_func(X):
    return np.expm1(X)

def make_log_transformer(func, inverse_func):
    log_transformer = FunctionTransformer(
        func=log_transform_func,
        inverse_func=inverse_log_transform_func,
        validate=True
    )
    return log_transformer

def make_preprocessor(transformer, transform_features):
    preprocessor = ColumnTransformer(
        transformers=[
            ('log_transform', transformer, transform_features)
        ],
        remainder='passthrough'
    )
    return preprocessor

def make_pipeline(preprocessor):
    pipeline = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, max_depth=30))
        ]
    )
    return pipeline

def make_model(pipeline, X_train, y_train):
    model = TransformedTargetRegressor(
        regressor=pipeline,
        func=np.log1p,
        inverse_func=np.expm1
    )
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
