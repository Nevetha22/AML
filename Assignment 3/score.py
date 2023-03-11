# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 14:22:27 2023

@author: Nevetha
"""
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer

filename = 'C:\\Users\\Nevetha\\AML\Assignment_02\\finalized_model.sav'
filename1 = 'C:\\Users\\Nevetha\\AML\Assignment_02\\vector_model.sav'
loaded_model = joblib.load(filename)
vect = joblib.load(filename1)

def score_model(text: str, model: BaseEstimator, threshold: float) -> tuple[bool, float]:
    # Convert the text to a numerical feature vector
    #text1 = [text]
    feature_vector=vect.transform([text])
    # summarize
    # encode document
    # Predict the class probability and the predicted class label
    class_prob = model.predict_proba(feature_vector)[0, 1]
    class_pred = class_prob >= threshold
    
    # Compute the propensity score as the ratio of the positive class examples in the training data
    pos_ratio = np.mean(model.classes_ == 1)
    propensity = class_prob / pos_ratio
    
    return class_pred, propensity

text_ham = "Hi, I'll be running late today"
text_spam = "dear Subscriber ur draw 4 £100 gift voucher with"

score_model(text_ham,loaded_model,0.3)
score_model(text_spam,loaded_model,0.1)

