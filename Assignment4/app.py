import os
import score
import pytest
import joblib
from textblob import TextBlob
from flask import Flask, jsonify, request
app = Flask(__name__)
loaded_model = joblib.load('finalized_model.sav')

@app.route('/score', methods=['GET','POST'])
def pred():
    os.system('pytest --cov=. --cov-report=term-missing --cov-report=html > coverage.txt &')
    return "Hello"

'''@app.route('/shutdown', methods=['POST'])
def pred1():
    return "Bye"'''

if __name__ == '__main__':
    app.run()