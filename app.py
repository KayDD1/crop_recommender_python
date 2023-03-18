import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd
from scipy.stats import boxcox

app=Flask(__name__)
# Load model from pickle
nb_model = pickle.load(open('nb_model.pkl', 'rb'))
scaled = pickle.load(open('scaled.pkl', 'rb'))


# Test prediction post request response with postman

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])

def predict_api():
    data = request.get_json(force=True)

    # boxcox(data["data"]["K_new"])
    data_df = pd.DataFrame(data['data'])
   
    data_df = scaled.transform(data_df)
    output=nb_model.predict(data_df)
    print(output[0])
    return jsonify(output[0])

# Create html input values prediction

@app.route('/predict', methods=['POST'])

def predict():
    if request.method == 'POST':
        # get the input values from html form
        K_new = request.form['K_new']
        N = request.form['N']
        P = request.form['P']
        Temperature = request.form['temperature']
        Rainfall = request.form['rainfall']
        Humidity = request.form['humidity']
        PH = request.form['ph']

        # convert the string values to float
        K_new = float(K_new)
        N = float(N)
        P = float(P)
        Temperature = float(Temperature)
        Rainfall = float(Rainfall)
        Humidity = float(Humidity)
        PH = float(PH)

        # create a numpy array of the input values
        input = np.array([[K_new, N, P, Temperature, Rainfall, Humidity, PH]]).reshape(1,7)

        final_input = scaled.transform(input)
   
        output=nb_model.predict(final_input)[0]
        return render_template("home.html",prediction_text="The crop type to plant is {}".format(output))


if __name__ == '__main__':
	app.run(port=5000, debug=True)
        







# # Define a function to get all class labels
# def get_labels(model):
#     labels = []
#     # Get the labels from the model
#     for i in range(len(model.classes_)):
#         labels.append(model.classes_[i])
#     # Return the list of labels
#     return labels

# # Create a Flask application
# from flask import Flask
# app = Flask(__name__)

# # Create a route for getting the class labels
# @app.route('/get_labels', methods=['GET'])
# def get_all_labels():
#     # Load the model
#     model = joblib.load('nb_model.pkl')
#     # Get the labels
#     labels = get_labels(model)
#     # Return the labels as JSON
#     return jsonify(labels)

# if __name__ == '__main__':
#     app.run()