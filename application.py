import pickle
import flask
import json
from flask import Flask, jsonify, request
import numpy as np
import xgboost
application = Flask(__name__)
#loading a model from a file called model.pkl
model =  pickle.load(open(filename,'rb'))
#model.load_model("model.json")

@application.route('/')
def hello_world():
    return 'Sup. Suboerzo'
