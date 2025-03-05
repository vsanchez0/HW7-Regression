"""
Write your unit tests here. Some tests to include are listed below.
This is not an exhaustive list.

- check that prediction is working correctly
- check that your loss function is being calculated correctly
- check that your gradient is being calculated correctly
- check that your weights update during training
"""
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv('./data/nsclc.csv')
X = df.select_dtypes(include=[np.number]).drop(columns=['NSCLC'], errors='ignore').values
y = df[['NSCLC']].values.ravel()

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

num_features = X_train.shape[1]
model = LogisticRegressor(num_feats=X_train.shape[1])


def test_prediction():
    y_pred = model.make_prediction(X_train)
    assert y_pred.shape == (X_train.shape[0],), "Prediction shape mismatch"
    assert np.all((y_pred >= 0) & (y_pred <= 1)), "Predicted values must be between 0 and 1"

def test_loss_function():
    y_pred = model.make_prediction(X_train)
    loss = model.loss_function(y_train, y_pred)
    assert isinstance(loss, float), "Loss should be a single float value"
    assert loss >= 0, "Loss should be non-negative"

def test_gradient():
    y_pred = model.make_prediction(X_train)
    gradient = model.calculate_gradient(y_train, X_train)
    assert gradient.shape == model.W.shape, "Gradient shape should match weights shape"
    assert np.all(np.isfinite(gradient)), "Gradient should not contain NaN or infinite values"

def test_training():
    initial_weights = model.W.copy()
    model.train_model(X_train, y_train, X_val, y_val)
    assert not np.array_equal(initial_weights, model.W), "Weights should update during training"