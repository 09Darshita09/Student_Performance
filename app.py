from flask import Flask, request, render_template
import pickle

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

app=Flask(__name__)

## Route for a home page 

model_file_path = 'artifacts/model.pkl'
with open(model_file_path, 'rb') as file:
    # Load the model
    model = pickle.load(file)

preprocessor_file_path = 'artifacts/preprocessor.pkl' 
with open(preprocessor_file_path, 'rb') as file:
    preprocessor = pickle.load(file)


@app.route('/')
def index():
    return render_template('home.html') 

@app.route('/',methods= ['POST','GET'])
def predict_datapoint():
    if request.method == 'POST':

        custom_data_input_dict = {
                "gender": [request.form.get('gender')],
                "race_ethnicity": [request.form.get('ethnicity')],
                "parental_level_of_education": [request.form.get('parental_level_of_education')],
                "lunch": [request.form.get('lunch')],
                "test_preparation_course": [request.form.get('test_preparation_course')],
                "reading_score": [request.form.get('reading_score')],
                "writing_score": [request.form.get('writing_score')],
            }
        
        pred_df =pd.DataFrame(custom_data_input_dict)
       
        print(pred_df)

        data_scaled = preprocessor.transform(pred_df)
        results = model.predict(data_scaled)
       
    return render_template('home.html',results=results[0])



if __name__ == '__main__':
    app.run(host="0.0.0.0",debug=True)