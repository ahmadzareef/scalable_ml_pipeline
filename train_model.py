# Script to train machine learning model.
import argparse
import json
import os
import pickle
import re
import sys

import joblib
import numpy
import pandas as pd

from data_file import process_data , load_data
from model_functions import train_test_split, inference, compute_model_metrics, train_model




def start(data):

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

    train, test = train_test_split(data)

    X_train, y_train, encoder, lb = process_data(train, categorical_features=cat_features, label="salary", training=True)
    # Train and save a model.
    lrc=train_model(X_train,y_train)
    joblib.dump(lrc, 'logistic_model.pkl')
    pickle.dump(encoder, open('encoder.pkl', 'wb'))
    pickle.dump(lb, open('lb.pkl', 'wb'))

    # testing the model
    X_test, y_test, _, _ = process_data(test, categorical_features=cat_features, label="salary", training=False,encoder=encoder, lb=lb)
    pred, pred_decoded = inference(X_test,lrc, lb)
    print(f"done train test {pred}, {pred_decoded}")
    pred_json = {"prediction": pred_decoded.tolist()}
    f = open("predictions.json", "w+")
    json.dump(pred_json, f)

    # compute metrics
    metrics = compute_model_metrics(y_test, pred)
    print(f"done compute metrics {metrics}")
    #dump metrics to json file

    with open("metrics.json", "w+") as f:
        json.dump(metrics,f)

    slice_data(data, cat_features, "salary")


    return lb,lrc,encoder,y_test

def slice_data(data,cat_features,label):
    json_metrics = {}
    for feature in cat_features:
      #  print(f"Slicing feature : {feature}")
      if feature == "sex":
        for cls in data[feature].unique():
            df_temp = data[data[feature] == cls]
           # json_metrics[feature]= feature
            encoder = pickle.load(open('encoder.pkl', 'rb'))
            lb = pickle.load(open('lb.pkl', 'rb'))
            lrc= joblib.load('logistic_model.pkl')
            X_test, y_test, _, _ = process_data(df_temp, categorical_features=cat_features, label=label,training=False, encoder=encoder, lb=lb)
            y_test_pred, y_pred_decode = inference(X_test,lrc, lb)
            metrics = compute_model_metrics(y_test, y_test_pred)
            json_metrics[feature+"_"+cls] = metrics
       # print(f"Slicing feature : {feature} is finished")

    with open("slice_output.json", "w+") as f:
        json.dump(json_metrics,f)



if __name__ == "__main__":


    # Add code to load in the data.
    data = load_data('census_mod.csv')
    start(data)
