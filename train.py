import os
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split

from data import process_data
from evaluate import evaluate_model
from model import compute_model_metrics
from model import inference
from model import train_model

DATA_FOLDER = "../data"
OUTPUT_FOLDER = "../output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

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


def main(dataset_name: str = "census_cleaned.csv", cat_cols: list = None):
    cat_cols = cat_cols or cat_features
    data_path = os.path.join(DATA_FOLDER, dataset_name)
    data = pd.read_csv(data_path)

    train_df, test_df = train_test_split(data, test_size=0.20)
    X_train, y_train, encoder, lb = process_data(
        train_df, categorical_features=cat_cols, label="salary", training=True
    )
    X_test, y_test, *_ = process_data(
        test_df,
        categorical_features=cat_cols,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )

    model = train_model(X_train, y_train)
    assets_path = "./saved_models/"
    assets = [model, encoder, lb]
    asset_filenames = ["trained_model.pkl", "encoder.pkl", "lb.pkl"]

    for name, asset in zip(asset_filenames, assets):
        with open(os.path.join(assets_path, name), "wb") as f:
            pickle.dump(asset, f)

    preds = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    evaluate_model(test_df, cat_cols, OUTPUT_FOLDER, model, encoder, lb)
    return model, precision, recall, fbeta


if __name__ == "__main__":
    _, precision, recall, fbeta = main()
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"Fbeta: {fbeta}")
