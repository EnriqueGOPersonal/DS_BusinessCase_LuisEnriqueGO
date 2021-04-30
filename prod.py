import pandas as pd
import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template
import os

### flask

app = Flask(__name__)
model = pickle.load(open('pipeline-model.pkl', 'rb'))

##leer feature values

featureDf = pd.read_csv("val_probabilities.csv")


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    idCliente = [int(x) for x in request.form.values()][0]
    clienteFeatures = featureDf[featureDf["ID_USER"] == idCliente].drop(["ID_USER", "fraude"], axis = 1)
    print(clienteFeatures)
    prediction = model.predict_proba(clienteFeatures)

    output = prediction[0][0]

    return render_template('index.html', prediction_text='Prediction {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)