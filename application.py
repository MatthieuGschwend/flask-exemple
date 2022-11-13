import pickle
import flask
from flask import Flask, jsonify, request
import xgboost
app = flask.Flask(__name__) 
import numpy as np
#loading a model from a file called model.pkl
#model = pickle.load(open("model.pkl","rb"))

@app.route('/')
def hello_world():
    return 'Sup. Suboerzo'
