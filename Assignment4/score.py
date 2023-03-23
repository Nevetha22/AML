#!/usr/bin/env python
# coding: utf-8

# In[4]:
import joblib
import sklearn
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer
# In[5]:
filename = 'C:\\Users\\Nevetha\\AML\\Assignment_3\\finalized_model.sav'
loaded_model = joblib.load(filename)
filename1 = 'C:\\Users\\Nevetha\\AML\\Assignment_3\\vector_model.sav'
vect = joblib.load(filename1)
# In[6]:
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