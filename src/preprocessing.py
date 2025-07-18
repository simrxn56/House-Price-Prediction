import pandas as pd
from sklearn.model_selection import train_test_split

def encode_categoricals(df):
    """
    Encode categorical features using one-hot encoding.
    Returns the encoded DataFrame.
    """
    return pd.get_dummies(df, drop_first=True)

def split_features_target(df, target_column):
    """
    Split DataFrame into features (X) and target (y).
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y

def split_train_test(X, y, test_size=0.25, random_state=12):
    """
    Split features and target into training and testing sets.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
