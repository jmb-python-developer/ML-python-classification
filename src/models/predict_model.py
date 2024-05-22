import copy, math
import numpy as np

def compute_gradient_vectorized_logistic(X, y, w, b, sigmoid):
    m = X.shape[0]  # number of examples
    # Compute predictions using sigmoid
    z = X.dot(w) + b
    predictions = sigmoid(z)
    # Errors: difference between predictions and actual values
    errors = predictions - y
    # Gradient w.r.t w
    dj_dw = X.T.dot(errors) / m
    # Gradient w.r.t b
    dj_db = np.sum(errors) / m
    
    return dj_dw, dj_db

def gradient_descent(X, y, w_in, b_in, alpha, num_iters, sigmoid):
    """
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
      
    Returns:
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w. 
      dj_db (scalar):       The gradient of the cost w.r.t. the parameter b. 
    """
    w = copy.deepcopy(w_in)
    b = b_in
    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient_vectorized_logistic(X, y, w, b, sigmoid)
        w = w - alpha * dj_dw
        b = b - alpha * dj_dw
    return w, b

def sigmoid(z):
  """
  Compute the Sigmoid for z function.

  Parameters:
    z: array
      Scalar or Numpy array of any size
  Returns:
    g: array
      sigmoid(z)
  """
  z = np.clip(z, -500, 500)
  g = 1.0 / (1.0 + np.exp(-z))
  return g
