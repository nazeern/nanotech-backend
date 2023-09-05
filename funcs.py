import numpy as np
from scipy.stats.mstats import gmean

# **********************
# Machine Learning Model Helpers
# **********************

def fit_log(X, y, noise=None):
    
    assert X.shape[0] == y.shape[0], "X dim and y dim must match"
    assert (noise is None) or (noise.shape[0] == X.shape[0]), "Must input noise weight for each data point"
    
    n = X.shape[0]
    
    if isinstance(noise, np.ndarray):
        D = np.diag(1 / noise ** 2)
    else:
        D = np.identity(n)
    
    X = np.log(X)
    X = np.vstack((X, np.ones(X.shape[0]))).T
    w_opt = np.linalg.solve(X.T @ D @ X, X.T @ D @ y)
    return w_opt

def predict_log(x, w):
    """
    General purpose logarithmic evaluator given weights w = [a, b]
    For our purposes, this function maps (concentration) ==> (impedance)
    
    Input: variable x, weights w = (a, b)
    Output: a * log(x) + b
    """
    a, b = w
    return a * np.log(x) + b

def predict_exp(x, w):
    """
    General purpose exponential evaluator given weights w = [a, b]
    This is the inverse of function y = a * log(x) + b
    For our purposes, this function maps (impedance) ==> (concentration)
    
    Input: variable x, weights w = (a, b)
    Output: exp( (y - b) / a )
    """
    a, b = w
    return np.exp( x / a - b / a , dtype=np.float128)

def fit(X, y, model='nitro'):
    """
    Fits a linear model to the data. If X is wide, attempts to fit via an RV decomposition.
    """
    m, n = X.shape
    if model == 'linear':
        if m < n:
            U, d, Vh = np.linalg.svd(X)
            R = U @ np.diag(d)
            V = Vh.T
            w_opt = V[:,:m] @ np.linalg.inv(R.T @ R) @ R @ y

        else:
            w_opt = np.linalg.inv(X.T @ X) @ X.T @ y
        
    elif model == 'nitro':
        assert y is not None and len(y) == m, "Must input valid concentrations"
        
        w = np.empty((n, 2))
        for i in range(n):
            w[i] = fit_log(np.array(y), X[:,i])
            
        return w
        
    return w_opt

def predict(X, w, model='nitro', agg=gmean, exclude_fns=0, return_preds=False):
    """
    Predict concentration given a list of impedances.
    """
    if model == 'linear':
        return X @ w
    
    elif model == 'nitro':
        
        assert agg is not None, "Must input valid aggregation function"
        
        n = len(X) # Number of elements in input vector
        
        preds = np.empty(n - exclude_fns)
        for j in range(w.shape[0] - exclude_fns):
            preds[j] = predict_exp(X[j], w[j])
    
        if return_preds:
            return agg(preds), preds
        else: 
            return agg(preds)
        
# **********************
# Data Processing
# **********************

def test_me():
    return str(np.pi)