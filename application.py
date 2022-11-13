#import pickle
import flask
from flask import Flask, jsonify, request
#import numpy as np
#import xgboost
app = flask.Flask(__name__) 
#loading a model from a file called model.pkl
#model = pickle.load(open("model.pkl","rb"))

@app.route('/')
def hello_world():
    return 'Sup. Suboerzo'
