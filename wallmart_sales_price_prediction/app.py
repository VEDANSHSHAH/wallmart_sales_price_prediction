from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np
import os

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'static/'
model_xgboost = pickle.load(open("models/model_xgboost_predictor.pkl", "rb"))
model_RandomForestRegressor = pickle.load(open("models/model_RandomForestRegressor_predictor.pkl", "rb"))
model_LinearRegression = pickle.load(open("models/model_LinearRegression_predictor.pkl", "rb"))

@app.route('/')
def home():
    pic1 = os.path.join(app.config['UPLOAD_FOLDER'], 'image.jpg')
    return render_template("index.html", image=pic1)

@app.route('/predict', methods=['GET', 'POST'])
def predict():    
    Store = int(request.form["Store"]) 
    Holiday_Flag = int(request.form["Holiday_Flag"])
    Temperature = float(request.form["Temperature"])
    Fuel_Price = float(request.form["Fuel_Price"])
    CPI = float(request.form["CPI"])
    Unemployment = float(request.form["Unemployment"]) 
    model_choice = int(request.form["model_choice"]) 
    input_data = (Store, Holiday_Flag,	Temperature, Fuel_Price, CPI, Unemployment)
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    if model_choice == 1:
        prediction = model_xgboost.predict(input_data_reshaped)[0]
    elif model_choice == 2:
        prediction = model_RandomForestRegressor.predict(input_data_reshaped)[0]
    elif model_choice == 3:
        prediction = model_LinearRegression.predict(input_data_reshaped)[0]

    prediction = round(prediction, 2)
    return str(prediction)

    
if __name__ == '__main__':
    app.run(debug=True) 