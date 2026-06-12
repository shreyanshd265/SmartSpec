from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipleine




application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            Rating=float(request.form.get('Rating')),
            Display=request.form.get('Display'),
            Generation=request.form.get('Generation'),
            Core=request.form.get('Core'),
            Ram=request.form.get('Ram'),
            SSD=request.form.get('SSD'),
            Warranty=request.form.get('Warranty'),
            Model=request.form.get('Model'),
            OS=request.form.get('OS'),
            Graphics=request.form.get('Graphics')
        )
        pred_df = data.get_dataframe()
        predict_pipeline = PredictPipleine()
        results = predict_pipeline.predict(pred_df)
        return render_template('home.html', results=round(float(results[0]), 2))

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=False)
