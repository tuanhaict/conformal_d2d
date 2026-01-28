import numpy as np
from typing import List, Callable

from utils.helpers import Grid, empirical_cdf, empirical_quantile
from utils.helpers import make_grid

def pava_weighted(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    n = len(y)
    v = y.copy()
    wv = w.copy()
    idx_start = np.arange(n)
    idx_end = np.arange(n)
    while True:
        viol = v[:-1] > v[1:]
        if not np.any(viol):
            break
        i = np.argmax(viol)
        w_new = wv[i] + wv[i+1]
        v_new = (wv[i]*v[i] + wv[i+1]*v[i+1]) / max(w_new, 1e-12)
        v[i] = v_new
        wv[i] = w_new
        idx_end[i] = idx_end[i+1]
        v = np.delete(v, i+1)
        wv = np.delete(wv, i+1)
        idx_start = np.delete(idx_start, i+1)
        idx_end = np.delete(idx_end, i+1)
        j = i
        while j > 0 and v[j-1] > v[j]:
            w_new = wv[j-1] + wv[j]
            v_new = (wv[j-1]*v[j-1] + wv[j]*v[j]) / max(w_new, 1e-12)
            v[j-1] = v_new
            wv[j-1] = w_new
            idx_end[j-1] = idx_end[j]
            v = np.delete(v, j)
            wv = np.delete(wv, j)
            idx_start = np.delete(idx_start, j)
            idx_end = np.delete(idx_end, j)
            j -= 1
    z = np.zeros(n)
    for val, s, e in zip(v, idx_start, idx_end):
        z[s:e+1] = val
    return z

def estimate_T_hat(X_list: List[np.ndarray], Y_list: List[np.ndarray], grid: Grid):
    m = len(grid.x)
    N = len(X_list)
    y_ij = np.zeros((N, m))
    w_ij = np.zeros((N, m))
    for i in range(N):
        Xi = X_list[i]
        Yi = Y_list[i]
        F_mu = empirical_cdf(Xi)
        Q_nu = empirical_quantile(Yi)
        p = F_mu(grid.x)
        y_ij[i] = Q_nu(p)
        counts, _ = np.histogram(Xi, bins=grid.bins)
        w_ij[i] = counts / max(len(Xi), 1)
    W_j = w_ij.sum(axis=0) + 1e-12
    y_bar = (w_ij * y_ij).sum(axis=0) / W_j
    z_hat = pava_weighted(y_bar, W_j)
    info = dict(y_ij=y_ij, w_ij=w_ij, W_j=W_j, y_bar=y_bar)
    return z_hat, info

def piecewise_linear_interp(xg: np.ndarray, zg: np.ndarray):
    def T(x):
        return np.interp(np.clip(x, 0.0, 1.0), np.clip(xg, 0.0, 1.0), np.clip(zg, 0.0, 1.0))
    return T