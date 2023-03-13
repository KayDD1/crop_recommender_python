import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd
from scipy.stats import boxcox

app=Flask(__name__)
## Load model from pickle
nb_model = pickle.load(open('nb_model.pkl', 'rb'))
scaled = pickle.load(open('scaled.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])

def predict_api():
    data = request.get_json(force=True)

    boxcox(data["data"]["K_new"])
    data_df = pd.DataFrame(data['data'])
   
    data_df = scaled.transform(data_df)
    output=nb_model.predict(data_df)
    print(output[0])
    return jsonify(output[0])

if __name__ == '__main__':
	app.run(port=5000, debug=True)

