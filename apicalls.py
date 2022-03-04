import ast
import datetime
import json
import os

import pandas as pd
import requests
from fastapi.testclient import TestClient

from main import app



def apicall():
    client = TestClient(app)
    with open('census_json.json') as json_file:
        json_data = json.load(json_file)
  #  print(json_data)
    with TestClient(app) as client:

        response = client.post('http://127.0.0.1:8000/predict/',json=json_data)
        print("Post Live")
        print(json_data)
        print(json.loads(response.text))

if __name__ == '__main__':
    apicall()