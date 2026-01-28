import math
import numpy as np
from random import *
import random
from sklearn import *
from dataclasses import dataclass
from typing import Callable, Tuple, List, Dict
import pickle, os
"""
returns indexed function from the cosine basis.
"""
def cosine_basis(index):

    def one(x):
        return 1

    def phi(x):
        return (2**.5) * math.cos(math.pi * index * x)

    if (not index): return one
    else: return phi

"""
c1, c2 are coefficient vectors. they should have the same length.
"""
def L2_distance(c1, c2):
    c1 = np.asarray(c1)
    c2 = np.asarray(c2)
    return np.linalg.norm(c1 - c2)


"""
i.e. Gaussian kernel function;
defined for x in [0, +inf)
"""
def RBF_kernel(x):
    x = np.asarray(x)
    return np.exp(-0.5 * x * x)

"""
Indices for basis functions in 6D.
There will be degree^6 index vectors.
"""
def alphas(degree):
    return [[a,b,c,d,e,f] for a in range(degree)
            for b in range(degree)
            for c in range(degree)
            for d in range(degree)
            for e in range(degree)
            for f in range(degree)]
def alphas_6D(degree):
    return [[a,b,c,d,e,f]
            for a in range(degree)
            for b in range(degree)
            for c in range(degree)
            for d in range(degree)
            for e in range(degree)
            for f in range(degree)]

def alphas_2D(degree):
    return [[a,b] for a in range(degree)
            for b in range(degree)]

def alphas_3D(degree):
    return [[a,b,c] for a in range(degree)
            for b in range(degree)
            for c in range(degree)]
def alphas_ND(dim):
    if dim == 2: return alphas_2D
    elif dim == 3: return alphas_3D
    elif dim == 6: return alphas_6D
    else: return None

def cosine_basis_ND(alpha, dim):

    def phi_alpha(x):
        result = 1
        for i in range(dim):
            phi_i = cosine_basis(alpha[i])
            result *= phi_i(x[i])
        return result

    return phi_alpha

"""
6D basis function corresponding to a given index vector (alpha).
"""
def cosine_basis_6D(alpha):

    # x should be a 6D vector of the form [a, b, c, d, e, f]
    def phi_alpha(x):
        result = 1
        for i in range(6):
            phi_i = cosine_basis(alpha[i])
            result *= phi_i(x[i])
        return result

    return phi_alpha

"""
List of all degree^6 fourier coefficients corresponding to a given
sample fit to num_terms. Coefficients are listed in the same order
as defined by alphas().
"""
def fourier_coeffs_6D(sample, degree):
    indices = alphas(degree)
    result = []
    for alpha in indices:
        phi_alpha = cosine_basis_6D(alpha)
        coeff = np.average([phi_alpha(s) for s in sample])
        result.append(coeff)
    return result

"""
List of all degree^N fourier coefficients corresponding to a given
sample fit to num_terms. Coefficients are listed in the same order
as defined by alphas().
"""
def fourier_coeffs_ND(sample, degree, dim):

    indices = alphas_ND(dim)(degree)
    result = []
    for alpha in indices:
        phi_alpha = cosine_basis_ND(alpha, dim)
        coeff = np.average([phi_alpha(s) for s in sample])
        result.append(coeff)
    return result

"""
Unbiased risk associated with fitting sample with a
nonparametric estimator with this degree
"""
def J_hat_ND(sample, degree, dim):

    indices = alphas_ND(dim)(degree)
    coeffs = fourier_coeffs_ND(sample, degree, dim)
    T = len(coeffs)
    n = len(sample)
    result = 0

    for j in range(T):
        alpha = indices[j]
        term = 0
        for i in range(n):
            term += (2./n) * ((cosine_basis_ND(alpha, dim)(sample[i]))**2 - (n + 1)*(coeffs[j])**2)
        result += term
    return (1./(n - 1))*result

def fourier_coeffs_ND_fast(sample, degree, dim):
    """
    sample: (N, dim)
    return: (num_coeffs,)
    """

    sample = np.asarray(sample)
    N = sample.shape[0]

    indices = alphas_ND(dim)(degree)  # list of tuples
    indices = np.array(indices)       # (M, dim)
    M = indices.shape[0]

    # Precompute cosines
    # cos_vals[i][k] = cos(pi * k * sample[:, i])
    cos_vals = []
    for i in range(dim):
        max_k = degree
        x = sample[:, i][:, None]              # (N,1)
        ks = np.arange(max_k + 1)[None, :]     # (1,K+1)
        cos_i = np.cos(math.pi * x * ks)        # (N, K+1)
        cos_vals.append(cos_i)

    coeffs = np.empty(M)

    for j, alpha in enumerate(indices):
        prod = np.ones(N)
        for i in range(dim):
            k = alpha[i]
            if k != 0:
                prod *= np.sqrt(2) * cos_vals[i][:, k]
        coeffs[j] = prod.mean()

    return coeffs
def fourier_coeffs(sample, t, nu=1., sigma=1.):
    """
    Vectorized version of fourier_coeffs.
    sample: 1D array-like of shape (n,)
    return: numpy array of shape (max_index+1,)
    """
    x = np.asarray(sample, dtype=float)
    n = x.size

    max_index = int(math.floor((t ** (1.0 / sigma)) / nu))

    coeffs = np.empty(max_index + 1, dtype=float)

    coeffs[0] = 1.0

    if max_index >= 1:
        ks = np.arange(1, max_index + 1, dtype=float)[:, None]  # (K,1)

        # x -> shape (1, n)
        x_row = x[None, :]  # (1,n)

        # phi_k(x_i) = sqrt(2) * cos(pi * k * x_i)
        vals = np.sqrt(2.0) * np.cos(np.pi * ks * x_row)  # (K,n)
        coeffs[1:] = vals.mean(axis=1)

    return coeffs
"""
actual estimated density function corresponding to nonparametric
estimation of the distribution from which sample was drawn,
computed directly from sample.

used to compute L1 distance, make plots.
"""
def approx_density(sample, t, nu =1. , sigma = 1.):

    coeffs = fourier_coeffs(sample, t, nu, sigma)

    def f_hat(x):
        return sum([coeffs[index]*cosine_basis(index)(x) for index in range(len(coeffs))])

    return f_hat

"""
actual estimated density function corresponding to nonparametric
estimation of the distribution from which sample was drawn,
computed from fourier coefficients.

used to make plots.
"""
# def coeffs_to_approx_density(coeffs):

#     def f_hat(x):
#         return sum([coeffs[index]*cosine_basis(index)(x) for index in range(len(coeffs))])

#     return f_hat

def coeffs_to_approx_density(coeffs):
    coeffs = np.asarray(coeffs, dtype=float)
    K = len(coeffs)

    def f_hat(x):
        x = np.asarray(x, dtype=float)

        # Handle scalar input
        scalar_input = False
        if x.ndim == 0:
            x = x[None]
            scalar_input = True

        ks = np.arange(K)[:, None]      # (K, 1)
        x_row = x[None, :]               # (1, N)

        basis = np.cos(np.pi * ks * x_row)
        if K > 1:
            basis[1:] *= np.sqrt(2.0)

        vals = (coeffs[:, None] * basis).sum(axis=0)

        return vals[0] if scalar_input else vals

    return f_hat


def empirical_cdf(x: np.ndarray):
    xs = np.sort(x)
    n = len(xs)
    def F(t):
        return np.searchsorted(xs, t, side="right") / n
    return F

def empirical_quantile(x: np.ndarray):
    xs = np.sort(x)
    n = len(xs)
    def Q(p):
        p = np.clip(p, 0.0, 1.0)
        idx = p * (n - 1)
        lo = np.floor(idx).astype(int)
        hi = np.ceil(idx).astype(int)
        w = idx - lo
        return (1-w) * xs[lo] + w * xs[hi]
    return Q

def push_forward_samples(x: np.ndarray, T: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    y = T(x)
    return np.clip(y, 0.0, 1.0)


def zeta_k(x: np.ndarray, k: int) -> np.ndarray:
    x = np.asarray(x)
    if k == 0:
        return x
    return x - np.sin(np.pi * k * x) / (abs(k) * np.pi)

def zeta_k_vec(u: np.ndarray, k: np.ndarray) -> np.ndarray:
    out = np.empty_like(u, dtype=float)
    mask0 = (k == 0)
    out[mask0] = u[mask0]
    mask = ~mask0
    out[mask] = u[mask] - np.sin(np.pi * k[mask] * u[mask]) / (np.abs(k[mask]) * np.pi)
    return out
def xi_map(a: np.ndarray, b : np.ndarray, k: np.ndarray, x: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    k = np.asarray(k, dtype=int)
    x = np.asarray(x, dtype=float)

    w = (2.0 * x - (a + b)) / (b - a)
    z = zeta_k_vec(w, k)
    y = 0.5 * (b - a) * z + 0.5 * (a + b)
    return y
def random_error_map(J: int = 6, k_support: List[int] = [1,2,3,4], p_k: List[float] = None) -> Callable[[np.ndarray], np.ndarray]:
    if p_k is None:
        p_k = np.ones(len(k_support), dtype=float) / len(k_support)

    Ks_abs = np.random.choice(k_support, size=J, p=p_k)
    signs = np.random.choice([-1, 1], size=J)
    Ks = (signs * Ks_abs).astype(int)


    U = np.sort(np.random.rand(J - 1))
    bounds = np.concatenate([[0.0], U, [1.0]])

    def T_eps(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)

        idx = np.searchsorted(bounds, x, side="right") - 1
        idx = np.clip(idx, 0, J - 1)

        a = bounds[idx]
        b = bounds[idx + 1]
        k = Ks[idx]

        y = xi_map(a, b, k, x)
        return y

    return T_eps

@dataclass
class Grid:
    x: np.ndarray
    bins: np.ndarray

def make_grid(m: int) -> Grid:
    bins = np.linspace(0.0, 1.0, m+1)
    x = 0.5 * (bins[:-1] + bins[1:])
    return Grid(x=x, bins=bins)

def quantile_from_samples(samples, q):
    """
    Stable quantile estimation for Wasserstein-1D.

    samples : array-like, shape (n_samples,)
    q       : quantile grid in (0,1), shape (m,)

    return  : quantile values Q(q), shape (m,)
    """
    samples = np.asarray(samples)
    samples = samples[np.isfinite(samples)]   # safety
    samples.sort()

    n = len(samples)
    if n == 0:
        raise ValueError("No valid samples.")

    # empirical CDF grid (midpoint rule)
    p = (np.arange(1, n + 1) - 0.5) / n

    # inverse CDF by interpolation
    return np.interp(q, p, samples)
def fourier_coeffs_quantile(Q, t, nu=1., sigma=1.):

    Q = np.asarray(Q, dtype=float)
    m = Q.size

    # ---- normalize to [0,1] (CRITICAL) ----
    q_min = Q.min()
    q_max = Q.max()
    Qn = (Q - q_min) / (q_max - q_min + 1e-12)

    # ---- same truncation rule as sample version ----
    max_index = int(math.floor((t ** (1.0 / sigma)) / nu))

    coeffs = np.empty(max_index + 1, dtype=float)

    # k = 0
    coeffs[0] = 1.0

    if max_index >= 1:
        ks = np.arange(1, max_index + 1, dtype=float)[:, None]  # (K,1)
        q_row = Qn[None, :]  # (1,m)

        vals = np.sqrt(2.0) * np.cos(np.pi * ks * q_row)  # (K,m)
        coeffs[1:] = vals.mean(axis=1)

    return coeffs
def load_data(name, num_data=1000, eta=1000, dim=3, file_path=None):
    scales = []
    if name == "mixture_of_betas":
        from utils.mixture_of_betas import generate_mixture_of_betas_data
        X_list, Y_list, _ = generate_mixture_of_betas_data(
            num_data=num_data,
            eta=eta,
            seed=42,
        )
    elif name == "trunc_gaussians":
        from utils.truncated_gaussians import generate_toy_mixture_data
        X_list, Y_list = generate_toy_mixture_data(
            num_data=num_data,
            eta=eta,
            seed=42,
        )
    elif name == "dark_matter":
        from utils.dark_matter import load_dark_matter_data
        X_list, Y_list = load_dark_matter_data(
            dim=dim,
            file_path=file_path
        )
    elif name == "house_price":
        from utils.house_price import load_house_price_data
        X_list, Y_list, scales = load_house_price_data(file_path=file_path)
    elif name == "mortality":
        from utils.mortality import load_mortality_data
        X_list, Y_list, scales = load_mortality_data(file_path=file_path)
    else:
        raise ValueError(f"Unknown dataset name: {name}")
    return X_list, Y_list, scales

def sample_ball_uniform_Lp(yi_hat, qhat, N, p=6, seed=None):
    """
    Uniform in L^p ball  { y : ||y - yi_hat||_p <= qhat }
    yi_hat : (d,)
    p      : scalar p >= 1  (here p = 6)
    """
    if seed is not None:
        np.random.seed(seed)

    yi_hat = np.asarray(yi_hat).ravel()
    d = yi_hat.shape[0]

    yk = np.zeros((N, d))

    for i in range(N):
        U = np.random.rand(d)
        signs = np.random.choice([-1, 1], size=d)
        X = signs * ( -np.log(U) )**(1.0/p )

        norm_p = np.sum(np.abs(X)**p)**(1.0/p)
        direction = X / (norm_p + 1e-12)
        u = np.random.rand()
        r = qhat * (u ** (1.0/d))
        yk[i] = yi_hat + r * direction

    return yk


def rejection_sample(xmin, xmax, pdf, count, batch_size=4096):
    xs = np.linspace(xmin, xmax, 2000)
    vals = pdf(xs)
    fn_max = np.max(vals)

    samples = []
    while len(samples) < count:
        # propose batch
        x = np.random.uniform(xmin, xmax, size=batch_size)
        h = np.random.uniform(0.0, fn_max, size=batch_size)

        # accept
        accept = h < pdf(x)
        samples.extend(x[accept])

    return np.asarray(samples[:count])