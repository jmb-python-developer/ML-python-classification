import copy, math
import numpy as np

def compute_gradient(X, y, w, b):
    """
    Args:
      X (ndarray (m,n)): Data, m examples with n features. A Matrix.
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
      
    Returns:
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w. 
      dj_db (scalar):       The gradient of the cost w.r.t. the parameter b. 
    """
    m,n = X.shape # m number of examples, n number of features (column)
    dj_db = 0.
    dj_dw = np.zeros((n,))
    for i in range(m):
      # err is the derivative of the biased or b parameter, per row (i)
      err = (np.dot(X[i], w) + b) - y[i]
      # Inner loop does the derivative per each 'w' parameter with respect to it's affecting 'x' feature (j)
      for j in range(n):
        dj_dw[j] = dj_dw[j] + err * X[i, j] 
      # Accumulates in outer loop updated b value, the partial derivative to db, per training data row and adds the value to total
      dj_db += err
    # How numpy works, this divides each parameter column by the training examples total 'm'
    dj_db = dj_db / m
    dj_dw = dj_dw / m
    return dj_db, dj_dw

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
