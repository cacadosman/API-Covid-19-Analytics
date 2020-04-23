import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import json
from flask import Flask
from flask import jsonify
from datetime import datetime
from datetime import timedelta

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

app = Flask(__name__)

def getData():
    r = requests.get("https://indonesia-covid-19.mathdro.id/api/harian")
    json_data = r.json()['data']
    if (json_data[-1]['jumlahKasusKumulatif'] == None):
        json_data.pop()
    data = eval(json.dumps(json_data).replace('null', 'None'))
    df = pd.DataFrame(data)
    return df[['harike']], df['jumlahKasusKumulatif'], df['tanggal']

@app.route('/api/v1/predict/', methods=['GET'])
def predict():
    X, y, dates = getData()
    model = make_pipeline(PolynomialFeatures(4), Ridge())
    model.fit(X, y)

    # Predict for next 5 days
    days = range(len(X)+1, len(X)+6)
    y_pred = model.predict([[i] for i in days])

    lastDate = datetime.fromtimestamp(list(dates)[-1]//1000)
    
    data = []
    for i in range(5):
        case = {}
        date = (lastDate + timedelta(days=(i+1))).strftime("%d/%m/%Y")
        case['date'] = date
        case['value'] = int(round(y_pred[i]))
        data += [case]    

    return jsonify(
        success=True,
        data=data
    )

if __name__ == '__main__':
    app.run(threaded=True, port=5000)