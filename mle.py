import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

from libfunc import compute_log_likelihood


def compute_params(mu_At, mu_Vt, std_A, std_V):
    """Compute the parameters for the MLE model from the free parameters.
        Equations from table I in 'Andersen, T. (2015). The early maximum 
        likelihood estimation model of audiovisual integration in speechperception'"""

    # get variances from standard deviations
    var_A = std_A ** 2
    var_V = std_V ** 2

    # reshape shifted means
    mu_At = mu_At.reshape(1, -1)
    mu_Vt = mu_Vt.reshape(-1, 1)

    # use equations from the paper to calculate mean and variance
    r_A, r_V = 1 / var_A, 1 / var_V
    w_A = r_A / (r_A + r_V)
    w_V = r_V / (r_A + r_V)
    mu_AV = w_A * mu_At + w_V * mu_Vt
    std_AV = 1 / np.sqrt(r_A + r_V)

    # return mean and standard deviation
    return mu_AV, std_AV


def compute_probs(c_A, c_V, std_A, std_V):
    # compute shifted means
    mu_At = np.arange(5) + 1 - c_A
    mu_Vt = np.arange(5) + 1 - c_V

    # compute parameters for audiovisual
    mu_AV, std_AV = compute_params(mu_At, mu_Vt, std_A, std_V)

    # compute the probabilities
    p_A = norm.cdf(mu_At, 0, std_A)
    p_V = norm.cdf(mu_Vt, 0, std_V)
    p_AV = norm.cdf(mu_AV, 0, std_AV)
    return p_A, p_V, p_AV


def objective_function(theta, data, n_samples):
    """ Compute MLE objective function on data (7, 5) from a single subject
        where theta=[c_A, c_V, log(std_A), log(std_V)]."""

    # extract parameters
    c_A, c_V, std_A, std_V = theta
    std_A, std_V = np.exp(std_A), np.exp(std_V)

    # compute probabilities
    p_A, p_V, p_AV = compute_probs(c_A, c_V, std_A, std_V)

    # compute and return the negative log-likelihood
    return compute_log_likelihood(p_A, p_V, p_AV, data, n_samples)


def fit(theta, data, n_samples):
    """Perform MLE fit to data for a single subject"""

    # optimize
    opt_result = minimize(objective_function, theta, args=(data, n_samples))

    # extract results
    objective = opt_result.fun
    c_A, c_V, std_A, std_V = opt_result.x

    return objective, c_A, c_V, std_A, std_V
