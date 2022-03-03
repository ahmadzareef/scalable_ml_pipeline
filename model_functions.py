import argparse
import json
import pickle

from data_file import load_data, process_data

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


# Optional: implement hyperparameter tuning.

def train_test(data):
    # data=da.load_data(path)
    train, test = train_test_split(data, test_size=0.20)
    return train, test


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    lrc = LogisticRegression()

    lrc.fit(X_train, y_train)

    return lrc


def inference(data , model,lb):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """

    pred = model.predict(data)

    pred_decoded = lb.inverse_transform(pred)
    # exists = True if "<=50K" or ">=50K" in pred_decoded else False
    # print(f"<=50K : {exists}")
    # print(f">=50K : {exists}")


    return pred, pred_decoded



def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    metrics= {'precision':precision, 'recall':recall, 'fbeta':fbeta}


    return metrics


if __name__ == "__main__":
    data = load_data('census_mod_without_label.csv')
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    lb = pickle.load(open('lb.pkl', 'rb'))
    model = joblib.load('logistic_model.pkl')
    encoder = pickle.load(open('encoder.pkl', 'rb'))
    X, _, _, _ = process_data(data, categorical_features=cat_features, label=None, training=False, encoder=encoder,
                              lb=lb)
    pred, pred_decoded=inference(X, model, lb)
    print(pred, pred_decoded)
    pred_json = {"prediction": pred_decoded.tolist()}
    f = open("predictions.json", "w+")
    json.dump(pred_json, f)

