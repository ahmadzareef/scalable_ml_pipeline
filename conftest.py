import pandas as pd
import pytest
from train_model import start

from data_file import process_data, load_data
from model_predict import *


@pytest.fixture(scope="session")
def data():
    data=load_data()
    return data

@pytest.fixture(scope="session")
def test_train_model(data):
    try:
        lb,model,encoder,y_test = start(data)

        # lb = pickle.load(open('lb.pkl', 'rb'))
        # model = joblib.load('logistic_model.pkl')
        # encoder = pickle.load(open('encoder.pkl', 'rb'))

        return lb,model,encoder,y_test

    except Exception as e:
        print(e)
        raise e


@pytest.fixture(scope="session")
def prepare_data_for_inf(test_train_model):
    lb, _, encoder, _ = test_train_model
    df = pd.read_csv('census_mod_without_label.csv')
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
    X, _, _, _ = process_data(df, categorical_features=cat_features, label=None, training=False, encoder=encoder,
                                  lb=lb)
    return X

@pytest.fixture(scope="session")
def test_inference(test_train_model,prepare_data_for_inf):
    try:
        lb,model,_,_ = test_train_model
        X = prepare_data_for_inf

        pred, pred_decoded = start_predict(X,model,lb)

        return pred, pred_decoded
    except Exception as e:
        print(e)
        raise e


#
#
# @pytest.fixture(scope="session")
# def test_compute_model_metrics(test_inference):
#     try:
#         pred, y_test = test_inference
#         metrice=compute_model_metrics(pred,y_test)
#         return  metrice
#     except Exception as e:
#         print(e)
#         raise e



# def pytest_addoption(parser):
#     parser.addoption("--data", action="store")


# @pytest.fixture(scope="session")
# def test_train_test(data):
#     try:
#         train, test = train_test(data)
#         return train, test ,data
#     except Exception as e:
#         print(e)
#         raise e


#
#@pytest.fixture(scope="session")
# def test_proc_train_test(test_train_test):
#     train ,test ,data = test_train_test
#     cat_features = [
#         "workclass",
#         "education",
#         "marital-status",
#         "occupation",
#         "relationship",
#         "race",
#         "sex",
#         "native-country",
#     ]
#     X_train, y_train, encoder, lb = process_data(
#         train, categorical_features=cat_features, label="salary", training=True
#     )
#
#     X_test, y_test, encoder, lb = process_data(test, categorical_features=cat_features, label="salary", training=False,
#                                                encoder=encoder, lb=lb)
#     return X_train, y_train, X_test, y_test, encoder, lb