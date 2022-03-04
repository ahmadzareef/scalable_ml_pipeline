import json
import pickle
import sys
from typing import List
from data_file import load_data, process_data

import joblib
from model_functions import inference , compute_model_metrics
import pandas as pd
from pydantic import BaseModel, Field
from train_model import start
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.encoders import jsonable_encoder
import os

for x in os.environ:
    print(x)

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    print("Check returned True")
    os.system("dvc config core.no_scm true")
    print("dvc config done")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed again!")
    try:
        os.system("rm -r .dvc .apt/usr/lib/dvc")
    except:
        print("deletion failed")


#################### declare Variables ####################
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



class jsondata(BaseModel):
    addage: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")

    class Config:
        schema_extra = {
            "example": {
                "addage": 0,
                "workclass": "string",
                "fnlgt": 0,
                "education": "string",
                "education-num": 0,
                "marital-status": "string",
                "occupation": "string",
                "relationship": "string",
                "race": "string",
                "sex": "string",
                "capital-gain": 0,
                "capital-loss": 0,
                "hours-per-week": 0,
                "native-country": "string"
            }
        }


app = FastAPI()

@app.get("/slice_output/")#,response_model=TaggedItem)
async def predict_check(
    filename: str = "slice_output.json"
):
    with open (filename,'r') as f:
        content = json.load(f)
    return content
@app.on_event("startup")
async def startup():
    print("Starting up")
    lb = pickle.load(open('lb.pkl', 'rb'))
    model = joblib.load('logistic_model.pkl')
    encoder = pickle.load(open('encoder.pkl', 'rb'))
    return lb  , model , encoder
# @app.post("/predict/")
# async def predict_check(
#     upload_file: UploadFile = File(..., description="A test_file read as UploadFile")
# ):
#     print(upload_file.filename)
#     dataframe = pd.read_csv(upload_file.file)
#     metriecs = start(dataframe)
#     output={"Prediction finished" :metriecs}
#     return output

@app.post("/predict/")
async def predict_json(
    data: List[jsondata]
):

    dataframe=pd.DataFrame(jsonable_encoder(data))
   # print(dataframe.head())
   #  lb = pickle.load(open('lb.pkl', 'rb'))
   #  model = joblib.load('logistic_model.pkl')
   #  encoder = pickle.load(open('encoder.pkl', 'rb'))
    lb , model , encoder = await startup()
    X, _, _, _ = process_data(dataframe, categorical_features=cat_features, label=None, training=False, encoder=encoder,
                              lb=lb)
    pred, pred_decoded = inference(X, model, lb)
    output=pred_decoded.tolist()
    return str(output)
#
# @app.post("/json_parser/")
# async def json_parser(
#     data: List[jsondata]
# ):
#
#     dataframe=pd.DataFrame(jsonable_encoder(data))
#     print(dataframe.head())
#     metriecs = start(dataframe)
#     output={"Prediction finished" :metriecs}
#     return output

#
# @app.post("/uploadfile/")
# async def create_upload_file(
#     upload_file: UploadFile = File(..., description="A test_file read as UploadFile") #,  model_train_predict :str = Form(...)
# ):
#     print(upload_file.filename)
#     output={"filename": upload_file.filename}
#     return output
#@app.get("/")
#async def main():
#    msg = "Welcome to the MLOps API"
#    return msg
@app.get("/upload_file")
async def main():
    msg = "Hello "
    content = """
<body>
<h1>Welcome ...</h1>
<h2>Model prediction</h2>
<form action="/predict/" enctype="multipart/form-data" method="post">

<input name="upload_file" type="test_file" multiple>
<input type="submit">
</form>

</body>
    """
    return HTMLResponse(content=content)
