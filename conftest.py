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


@pytest.fixture()
def below_50k_example():
    return [{
        "addage": 45,
        "workclass": "State-gov",
        "fnlgt": 2334,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Prof-specialty",
        "relationship": "Wife",
        "race": "Black",
        "sex": "Female",
        "capital-gain": 2174,
        "capital-loss": 0,
        "hours-per-week": 60,
        "native-country": "Cuba",
    }]


@pytest.fixture()
def above_50k_example():
    return [{
        "addage": 52,
        "workclass": "Self-emp-not-inc",
        "fnlgt": 209642,
        "education": "Bachelors",
        "education-num": 9,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 123387,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States",
    }]