import pickle

from flask import Flask, jsonify, request
import numpy as np
import xgboost
application = Flask(__name__)
#loading a model from a file called model.pkl
#model = pickle.load(open("model.pkl","rb"))

@application.route('/')
def hello_world():
    return 'Sup. Suboerzo'
