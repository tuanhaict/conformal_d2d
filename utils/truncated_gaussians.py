import numpy as np
from scipy.stats import truncnorm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy as np
def sample_trunc_gaussian(mu, sig, size):
    """
    Sample from N(mu, sig^2) truncated to [0, 1]
    """
    a = (0.0 - mu) / sig
    b = (1.0 - mu) / sig
    return truncnorm.rvs(a, b, loc=mu, scale=sig, size=size)

def sample_mixture_2_trunc_gaussians(mu1, mu2, sig1, sig2, size):
    """
    0.5 * TruncN(mu1, sig1) + 0.5 * TruncN(mu2, sig2)
    """
    z = np.random.rand(size) < 0.5
    samples = np.empty(size)

    n1 = z.sum()
    n2 = size - n1

    if n1 > 0:
        samples[z] = sample_trunc_gaussian(mu1, sig1, n1)
    if n2 > 0:
        samples[~z] = sample_trunc_gaussian(mu2, sig2, n2)

    return samples

def generate_toy_mixture_data(num_data, eta, seed=None):
    """
    Parameters
    ----------
    num_data : int
        Number of dataset instances
    eta : int
        Number of samples per distribution
    seed : int or None
        Random seed

    Returns
    -------
    input_datas  : np.ndarray, shape (num_data, eta)
    output_datas : np.ndarray, shape (num_data, eta)
    """

    if seed is not None:
        np.random.seed(seed)

    # Draw parameters
    # mu_1, mu_2 ~ Unif[0, 1]
    mus = np.random.rand(num_data, 2)

    # sig_1, sig_2 ~ Unif[0.05, 0.10]
    sigs = 0.05 * (1.0 + np.random.rand(num_data, 2))

    input_datas = np.empty((num_data, eta))
    output_datas = np.empty((num_data, eta))

    for i in range(num_data):
        mu1, mu2 = mus[i]
        sig1, sig2 = sigs[i]

        # p(x)
        input_datas[i] = sample_mixture_2_trunc_gaussians(
            mu1, mu2, sig1, sig2, eta
        )

        # q(x): reflected version
        output_datas[i] = sample_mixture_2_trunc_gaussians(
            1.0 - mu1, 1.0 - mu2, sig1, sig2, eta
        )

    return input_datas, output_datas
