from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np
import json

data = pd.read_csv("New_Bangalore_houses_prices.csv")
app = Flask(__name__)
pipe = pickle.load(open("./LrModel.pkl",'rb'))

@app.route('/')
def Index():
    locations = sorted(data['location'].unique())
    return render_template("BHPP_index.html", locations = locations)

@app.route('/Predict', methods=['Post','GET'])
def predict():
    data = request.get_json()
    print(data)
    input = pd.DataFrame([data])
    
    prediction = pipe.predict(input)[0] * 1e5

    return str(np.round(prediction,2))

if __name__=='__main__':
    app.run(debug=True, port=5001)