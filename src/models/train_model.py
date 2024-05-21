import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath('../../'))

from src.models.predict_model import sigmoid

def compute_cost_logistic_sq_err(X, y, w, b):
    '''
    The function computes the cost (logistic squared error) for the logistic regression model with parameters w 
    and b using the data X and y.

    Args:
      X (ndarray): Shape (m,n) matrix of examples with multiple features
      w (ndarray): Shape (n)   parameters for prediction
      b (scalar):              parameter  for prediction, bias
    Returns:
      cost (scalar): cost
    '''
    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        # Calculate the linear predicted value for z at given row and with 'w' params (vector) and 'b' bias (scalar)
        z_i = np.dot(X[i], w) + b
        # Calculate the logistic predicted value for the previous row
        f_wb_i = sigmoid(z_i)
        # Increase the accumulated cost
        cost += (f_wb_i - y[i])**2
    # To comply with logistic squared error, diving by 2m
    cost /= 2 * m
    return np.squeeze(cost)
