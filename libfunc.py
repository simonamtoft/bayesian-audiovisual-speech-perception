import numpy as np
from scipy.special import comb
from scipy.stats import binom


def binomial_pmf(k, n, p):
    """Calculate the Binomial pmf"""
    return comb(n, k) * np.power(p, k) * np.power(1 - p, n - k)


def compute_log_likelihood(p_A, p_V, p_AV, data, n_samples):
    probs = np.vstack([p_A.T, p_V.T, p_AV])
    L = binom.logpmf(data, n_samples, probs).sum() 
    # L = np.log(binomial_pmf(data, n_samples, probs)).sum() # This has numerical issues but it works on the datasets :-)
    return -L


def to_table_body(matrix):
    return "\\\\\n".join([
        f"Subject {i+1} & " + " & ".join(np.char.mod('%.2f', x)) 
        for i, x in enumerate(matrix)
    ])
