import numpy as np
from dataclasses import dataclass

from utils.helpers import make_grid, push_forward_samples, random_error_map, zeta_k


@dataclass
class BetaMixtureParams:
    alphas: np.ndarray
    betas: np.ndarray
    pis: np.ndarray

def sample_beta_mixture(n: int, params: BetaMixtureParams) -> np.ndarray:
    z = np.random.choice(3, size=n, p=params.pis)
    x = np.empty(n)
    for j in range(3):
        mask = (z == j)
        if np.any(mask):
            a = params.alphas[j]
            b = params.betas[j]
            x[mask] = np.random.beta(a, b, size=mask.sum())
    return x

def random_beta_mixture_params() -> BetaMixtureParams:
    alphas = np.random.uniform(1.0, 10.0, size=3)
    betas  = np.random.uniform(1.0, 10.0, size=3)
    pis = np.array([0.3, 0.4, 0.3])
    pis = pis / pis.sum()
    return BetaMixtureParams(alphas=alphas, betas=betas, pis=pis)

def generate_mixture_of_betas_data(num_data: int, eta: int, seed: int = None):
    if seed is not None:
        np.random.seed(seed)
    def T0(x): return zeta_k(np.asarray(x), 4)
    X_list, Y_list, params_list = [], [], []
    for i in range(num_data):
        params = random_beta_mixture_params()
        params_list.append(params)
        Xi = sample_beta_mixture(eta, params)
        T_eps = random_error_map(20, k_support=[1,2,3,4])
        Yi = push_forward_samples(push_forward_samples(Xi, T0), T_eps)
        X_list.append(Xi)
        Y_list.append(Yi)
    return X_list, Y_list, params_list