import pickle
import numpy as np
import pandas as p
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, jsonify, render_template

application = Flask(__name__)
app = application

## import ridge regressor and standard scaler pickle

ridgeModel = pickle.load(open("models/ridgereg.pkl", "rb"))
standardScaler = pickle.load(open("models/scaler.pkl", "rb"))


@app.route("/")       # Home page
def home():
    return render_template("index.html")

@app.route("/predictData", methods = ["GET", "POST"])
def predict_datapoint():
    if (request.method == "POST"):
        # Example: get values from form
            data = [float(x) for x in request.form.values()]
            
            # Convert to array
            input_data = np.array([data])
            
            # Scale
            scaled_data = standardScaler.transform(input_data)
            
            # Predict
            result = ridgeModel.predict(scaled_data)
            
            return render_template("home.html", result=result[0])
    else :
        return render_template("home.html")

if __name__ == "__main__" :
    app.run(debug = True)