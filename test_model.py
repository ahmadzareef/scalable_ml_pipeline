from unittest.mock import ANY

import pandas as pd
import pytest
from pytest_mock import mocker 

import model as cmm


@pytest.fixture()
def X_train():
    return pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]])


@pytest.fixture()
def y_train():
    return pd.Series([0, 1, 1])


@pytest.fixture()
def y():
    return y_train


@pytest.fixture()
def preds():
    return y_train


@pytest.fixture()
def test_model(preds):
    class MockModel:
        def __init__(self, preds=preds):
            self.preds = preds

        def predict(self, X):
            return self.preds

    return MockModel()



def test_train_model(mocker, X_train, y_train):
    mock_model = mocker.patch("model.SVC")

    _ = cmm.train_model(X_train, y_train)

    mock_model.assert_called()
    mock_model().fit.assert_called_once_with(X_train, y_train)



def test_compute_model_metrics(mocker, y, preds):
    mock_fbeta_score = mocker.patch("model.fbeta_score")
    mock_precision_score = mocker.patch("model.precision_score")
    mock_recall_score = mocker.patch("model.recall_score")

    _ = cmm.compute_model_metrics(y, preds)

    mock_fbeta_score.assert_called_once_with(y, preds, beta=ANY, zero_division=1)
    mock_precision_score.assert_called_once_with(y, preds, zero_division=1)
    mock_recall_score.assert_called_once_with(y, preds, zero_division=1)


def test_inference(test_model, X_train, preds):
    inference_output = cmm.inference(test_model, X_train)
    assert inference_output == preds
