import json

import pandas as pd
import requests


with open('census1.json') as json_file:
    json_data = json.load(json_file)

dataframe = pd.DataFrame(json_data)
print([x for x in dataframe.columns])
response = requests.post('https://udacity-mlops-fastapis.herokuapp.com/predict/',json=json_data)
print(response.text)
print(response.status_code)
