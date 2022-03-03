import json
import pickle
import joblib
from data_file import process_data , load_data
from model_functions import  inference

def start_predict(data, model=None, lb=None):

   # lb = pickle.load(open('lb.pkl', 'rb'))
  #  model = joblib.load('logistic_model.pkl')
    pred, pred_decoded=inference(data, model, lb)

    return pred, pred_decoded


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
    model = joblib.load('logistic_model.pkl')
    lb = pickle.load(open('lb.pkl', 'rb'))
    encoder = pickle.load(open('encoder.pkl', 'rb'))
    X, _, _, _ = process_data(data, categorical_features=cat_features, label=None, training=False, encoder=encoder,
                              lb=lb)
    pred, pred_decoded = start_predict(X,model,lb)
    print(pred, pred_decoded)
    pred_json = {"prediction": pred_decoded.tolist()}
    f = open("predictions.json", "w+")
    json.dump(pred_json, f)