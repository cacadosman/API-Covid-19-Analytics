import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import json
from flask import Flask
from flask import jsonify

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

app = Flask(__name__)

def getData():
    r = requests.get("https://indonesia-covid-19.mathdro.id/api/harian")
    json_data = r.json()['data']
    df = pd.DataFrame(eval(json.dumps(json_data)))
    return df[['harike']], df['jumlahKasusKumulatif']

@app.route('/api/v1/predict/', methods=['GET'])
def predict():
    X, y = getData()
    model = make_pipeline(PolynomialFeatures(4), Ridge())
    model.fit(X, y)

    # Predict for next 3 days
    days = range(len(X)+1, len(X)+4)
    y_pred = model.predict([[i] for i in days])

    data = {}
    data['from'] = days[0]
    data['to'] = days[-1]
    data['values'] = [int(i) for i in y_pred]

    return jsonify(
        success=True,
        data=data
    )

if __name__ == '__main__':
    app.run(threaded=True, port=5000)