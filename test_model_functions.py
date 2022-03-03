import json
import pickle

import joblib
import numpy
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder

import model_functions
from train_model import start

def testing_train_model(test_train_model):
    try:
        lb,model,encoder,_ = test_train_model

        assert type(lb) == LabelBinarizer
        assert type(model) == LogisticRegression
        assert type(encoder) == OneHotEncoder
    except Exception as e:
        print(e)
        raise e

#
def testing_inference( test_inference):
    try:
        pred, pred_decoded = test_inference

        assert type(pred) == numpy.ndarray
        assert type(pred_decoded) == numpy.ndarray
    except Exception as e:
        print(e)
        raise e

def testing_compute_model_metrics():
    try:
        metrice=json.load(open("metrics.json",'rb'))
        assert type(metrice["precision"]) == float
        assert type(metrice["recall"]) == float
        assert type(metrice["fbeta"]) == float
        assert metrice["precision"] > 0.5
        assert metrice["recall"] > 0.2
        assert metrice["fbeta"] > 0.35
        print (metrice)
    except Exception as e:
        print(e)
        raise e

def testing_data_slices():
    try:
        data_slices = json.load(open("slice_output.json"))

        # cat_features = [
        #     "workclass",
        #     "education",
        #     "marital-status",
        #     "occupation",
        #     "relationship",
        #     "race",
        #     "sex",
        #     "native-country",
        # ]
        for key in data_slices:
            #for feature in cat_features:
            #    if  key == "sex" :
                    precision=(data_slices[key]["precision"])
                    recall=(data_slices[key]["recall"])
                    fbeta=(data_slices[key]["fbeta"])
                    assert type(precision) == float
                    assert type(recall) == float
                    assert type(fbeta) == float
    except Exception as e:
        print(e)
        raise e

