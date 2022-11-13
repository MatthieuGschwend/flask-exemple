import pickle

from flask import Flask, jsonify, request
import numpy as np
import xgboost
application = Flask(__name__)
#loading a model from a file called model.pkl
model = pickle.load(open("model.pkl","rb"))

@application.route('/')
def hello_world():
    return 'Sup. Suboerzo'

@application.route('/okpath')
def hello_world2():
    return 'okpath'

@application.route('/predict', methods=['POST','GET'])
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
 #return query_parameters['feature_array']
