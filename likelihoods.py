import numpy as np
from scipy.special import comb


def binomial_pmf(k, n, p):
    """Calculate the Binomial pmf"""
    return comb(n, k) * np.power(p, k) * np.power(1 - p, n - k)


def compute_log_likelihood(p_A, p_V, p_AV, data, n_samples):
    probs = np.vstack([p_A.T, p_V.T, p_AV])
    L = np.log(binomial_pmf(data, n_samples, probs)).sum()
    return -L

