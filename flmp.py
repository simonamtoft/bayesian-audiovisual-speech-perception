import numpy as np
from scipy.optimize import minimize

from libfunc import compute_log_likelihood


def baseline_softmax(x):
    """Calculate softmax of x with a baseline 0, such that result has one more parameter."""
    x = np.concatenate([
        np.array([x]).flatten(), [0]
    ])
    e = np.exp(x)
    return e / e.sum()


def compute_probs(theta_A, theta_V):
    # compute audio and visual probabilities
    p_A = np.array([baseline_softmax(t)[0] for t in theta_A]).reshape(-1,1)
    p_V = np.array([baseline_softmax(t)[0] for t in theta_V]).reshape(-1,1)

    # compute audiovisual probabilities by
    # taking the outer product for all combinations of audio and visual
    p_AV = (p_A @ p_V.T) / (p_A @ p_V.T + (1 - p_A) @ (1 - p_V).T)

    return p_A, p_V, p_AV


def objective_function(theta, data, K, n_samples=24):
    # extract audio and visual parameters
    theta_A = theta[0:K]
    theta_V = theta[K: ]

    # compute probabilities from parameters
    p_A, p_V, p_AV = compute_probs(theta_A, theta_V)
    
    # compute and return the negative log-likelihood    
    return compute_log_likelihood(p_A, p_V, p_AV, data, n_samples)


def fit(theta, data, n_samples, K):
    """Perform FLMP fit to data"""
    opt_result = minimize(objective_function, theta, args=(data, K, n_samples))
    objective, theta_A, theta_V = (
        opt_result.fun, 
        (opt_result.x[0:K]), 
        (opt_result.x[K:])
    )
    return objective, theta_A, theta_V