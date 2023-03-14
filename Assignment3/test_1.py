import os
import score
import joblib
import pytest
import requests

loaded_model = joblib.load('finalized_model.sav')

cases = ["Kindly settle the dues shown in the statement immediately",
         "Hi Customer, here is a chance to win $10,000?",
         "dear Subscriber here is the chance to draw 4 £100 gift voucher with just one click",
         "Let's have the meeting at 10AM",
         "Free gift worth $5000 follow the link for more details"]

cases1 = {"Kindly settle the dues shown in the statement immediately":False,
         "Hi Customer, here is a chance to win $10,000?":True,
         "dear Subscriber here is the chance to draw 4 £100 gift voucher with just one click":True,
         "Let's have the meeting at 10AM":False,
         "Free gift worth $5000 follow the link for more details":True}

#Smoke Test
def test_smoke():
        for case in cases:
            s = score.score_model(case, loaded_model,0.2)
            assert s[0] == cases1[case]

# Test if the prediction is either True or False(True=Spam)
def test_pred():
        for case in cases:
            s = score.score_model(case, loaded_model,0.2)[0]
            assert s in [True, False]

# Test if the propensity score is between 0 to 1
def test_prop():
        for case in cases:
            s = score.score_model(case, loaded_model,0.2)[1]
            assert 0<= s <=1  

# Test if prediction is True if threshold is 0 and if it is False if threshold is 1
def test_prop_0and1():
        for case in cases:
            s = score.score_model(case, loaded_model,0)[0]
            assert s == True
        for case in cases:
            s = score.score_model(case, loaded_model,1)[0]
            assert s == False

# Tesing with an obvious spam input
def test_obv_spam():
        s = score.score_model(cases[2], loaded_model,0.2)[0]
        assert s == True

# Testing with an obvious non-spam input
def test_obv_ham():
        s = score.score_model(cases[0], loaded_model,0.2)[0]
        assert s == False

def main():
    test_flask()

# Integration test
def test_flask():
    # Test cases
    response = requests.post('http://127.0.0.1:5000/score')
    # Close the Flask app
    response = requests.post('http://127.0.0.1:5000/shutdown')

if __name__ == "__main__":
    main()
    