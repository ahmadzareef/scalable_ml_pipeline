import json
import os
import sys
from http import HTTPStatus
from pathlib import Path

import pandas as pd
import requests
from fastapi.testclient import TestClient

from main import app
import sys

sys.path.append('.')
client = TestClient(app)


def test_get():
    with TestClient(app) as client:
        response= client.get("http://127.0.0.1:8000/")

        assert response.status_code == 200
        assert response.text == '"Welcome to the MLOps API"'

def test_slice_output():
    with TestClient(app) as client:
        response= client.get("http://127.0.0.1:8000/slice_output")
    with open ("slice_output.json",'r') as f:
        content = json.load(f)

    response_data=json.loads(response.text)
    print(len(response_data))
    assert response.status_code == 200
    assert len(response_data) == len(content)


def predictions_cats():
    with open('census_json.json') as json_file:
        json_data = json.load(json_file)

    with TestClient(app) as client:
        response = client.post('http://127.0.0.1:8000/predict/',json=json_data)

    jdata=response.text
    for i in jdata.split(','):

        assert i == '<=50K'
        print(i)
    assert response.status_code == 200


def test_json():
    with open('census_json.json') as json_file:
        json_data = json.load(json_file)

    with TestClient(app) as client:
        response = client.post('http://127.0.0.1:8000/predict/',json=json_data)

    jdata=response.text
    for i in jdata.split(','):
        #assert i != ''
        assert i.replace('[','').replace(']','').replace('\'','') == '"<=50K"'

    assert response.status_code == 200
