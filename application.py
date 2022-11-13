import pickle
import flask
from flask import Flask, jsonify, request
app = flask.Flask(__name__) 
import numpy as np
#loading a model from a file called model.pkl
model = pickle.load(open("model.pkl","rb"))


@app.route('/predict', methods=['POST'])
def predict():

 #getting an array of features from the post request's body
 query_parameters = request.args
 feature_array = np.fromstring(query_parameters['feature_array'],dtype=float,sep=",")

 #creating a response object
 #storing the model's prediction in the object
 response = {}
 response['predictions'] = model.predict_proba([feature_array]).tolist()

 #returning the response object as json
 return flask.jsonify(response)
if __name__ == "__main__":
 app.run(debug=True)