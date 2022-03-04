import os

import pandas as pd
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pydantic import Field

from data import process_data
from model import inference
from train import cat_features
from utils import load_asset

if "DYNO" in os.environ and os.path.isdir("./.dvc"):
    print("Running DVC")
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r ./.dvc ./apt/usr/lib/dvc")
else:
    print("dvc operation not performed")

app = FastAPI()


def hyphen_to_underscore(field_name):
    return f"{field_name}".replace("_", "-")


class Input(BaseModel):
    age: int = Field(..., example=45)
    capital_gain: int = Field(..., example=2174)
    capital_loss: int = Field(..., example=0)
    education: str = Field(..., example="Bachelors")
    education_num: int = Field(..., example=13)
    fnlgt: int = Field(..., example=2334)
    hours_per_week: int = Field(..., example=60)
    marital_status: str = Field(..., example="Never-married")
    native_country: str = Field(..., example="Cuba")
    occupation: str = Field(..., example="Prof-specialty")
    race: str = Field(..., example="Black")
    relationship: str = Field(..., example="Wife")
    sex: str = Field(..., example="Female")
    workclass: str = Field(..., example="State-gov")

    class Config:
        alias_generator = hyphen_to_underscore
        allow_population_by_field_name = True


@app.get("/")
async def root():
    return JSONResponse(
        status_code=200,
        content={"message": "Welcome to App! Let's make some predictions :)"},
    )


@app.post("/model/")
async def predict(data: Input):
    model = load_asset("trained_model.pkl")
    encoder = load_asset("encoder.pkl")
    lb = load_asset("lb.pkl")

    # TODO: Modify to pass in multiple indices
    df = pd.DataFrame(data.dict(by_alias=True), index=[0])
    X, *_ = process_data(
        df, categorical_features=cat_features, label=None, encoder=encoder, lb=lb, training=False
    )
    predictions = inference(model, X)
    return JSONResponse(status_code=200, content=predictions.tolist())
