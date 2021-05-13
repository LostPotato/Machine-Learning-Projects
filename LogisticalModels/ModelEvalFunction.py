import numpy as np


"""

This file is just for the purpose of writing some manual functions for logistic models in order
to understand the inner workings of the model

"""


def log_odds(features, coefficients, intercept):
    """
  Returns the log odds of the model by taking the dot product of the features and coefficients
  """
    return np.dot(features, coefficients) + intercept


def sigmoid(z):
    """
    Defines the sigmond function of the logistical model by taking the negative exp. of the log odds
    """
    denominator = 1 + np.exp(-z)
    return 1 / denominator


# Create predict_class() function here
def predict_class(features, coefficients, intercept, threshold):
    """
  Classifing the data based on a threshold for a logistical regression model
  sigmoid function classification. TL:DR defines the sensitive of the classification model
  by establishing the threshold
  """
    calculated_log_odds = log_odds(features, coefficients, intercept)
    probabilities = sigmoid(calculated_log_odds)
    return np.where(probabilities >= threshold, 1, 0)
