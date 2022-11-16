import pickle
import flask
import json
from flask import Flask, jsonify, request
import numpy as np
import xgboost
app = Flask(__name__)
#loading a model from a file called model.pkl
model =  pickle.load(open('model.pkl','rb'))
#model.load_model("model.json")

@app.route('/')
def hello_world():
    return 'Afficher simplement un message'


@app.route('/okpath')
def hello_world2():
    print('randomstr&')
    print(type(model))
    return type(model)

@app.route('/predict', methods=['POST','GET'])
def predict():

 #getting an array of features from the post request's body
 #query_parameters = request.args
 #feature_array = np.fromstring(query_parameters['feature_array'],dtype=float,sep=",")
 feature_array = [ 1.00000000e+00,  0.00000000e+00,  0.00000000e+00,  2.00000000e+00,
        2.00000000e+00,  1.00000000e+00, -9.46100000e+03, -2.12000000e+03,
        0.00000000e+00,  0.00000000e+00,  1.00000000e+00, -1.13400000e+03,
        2.62948593e-01,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  7.50000000e-01,  2.50000000e-01,
       -1.43700000e+03, -1.03000000e+02, -8.74000000e+02, -4.99875000e+02,
       -6.37000000e+02,  1.39375780e-01, -1.43700000e+03, -4.76000000e+02,
       -9.74500000e+02, -6.61333333e+02]

 #creating a response object
 #storing the model's prediction in the object
 response = {}
 response['predictions'] = model.predict_proba([feature_array]).tolist()

 #returning the response object as json
    
 return flask.jsonify(response)
 #return query_parameters['feature_array']
