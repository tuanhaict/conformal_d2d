import argparse
from math import ceil
import math
import time

from matplotlib import pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors

from models.nonparametric import Estimator
from utils.helpers import L2_distance, fourier_coeffs, load_data, push_forward_samples, random_error_map, zeta_k
from utils.mixture_of_betas import random_beta_mixture_params, sample_beta_mixture
from utils.truncated_gaussians import generate_toy_mixture_data
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default="mixture_of_betas",
        help="Dataset to use",
    )
    parser.add_argument(
        "--num_data",
        type=int,
        default=3200,
        help="Total number of data samples",
    )
    parser.add_argument(
        "--eta",
        type=int,
        default=1000,
        help="Number of samples per distribution",
    )
    parser.add_argument(
        "--truncation",
        type=int,
        default=10,
        help="Truncation level for Fourier coefficients",
    )
    parser.add_argument(
        "--num_trains",
        type=int,
        default=1000,
        help="Number of training samples",
    )
    parser.add_argument(
        "--num_cals",
        type=int,
        default=200000,
        help="Number of calibration samples",
    )
    parser.add_argument(
        "--num_cvs",
        type=int,
        default=200,
        help="Number of cross-validation samples",
    )
    parser.add_argument(
        "--k_neighbors",
        type=int,
        default=500,
        help="Number of neighbors for adaptive CP"
    )
    opt = parser.parse_args()
    return opt
def calculate_conditional_times(cal_x_coeffs, K, xs, ys_hat, cal_scores, alpha, t):
    nn = NearestNeighbors(
        n_neighbors=K,
        metric="euclidean",
        algorithm="brute",
    )
    nn.fit(cal_x_coeffs)
    start_time = time.time()
    for h in range(len(xs)):
        x = xs[h]
        y_hat = ys_hat[h]
        x_coeffs = np.asarray(
            fourier_coeffs(x, t)
        ).reshape(1, -1)

        distances, indices = nn.kneighbors(
            x_coeffs,
            n_neighbors=K,
            return_distance=True,
        )

        neighbor_idx = indices[0]
        local_scores = cal_scores[neighbor_idx]
        sorted_scores = np.sort(local_scores)

        # Quantile conformal
        k_star = int(math.ceil((K + 1) * (1 - alpha))) - 1
        k_star = max(0, min(k_star, K - 1))
        q_alpha = sorted_scores[k_star]
        ys_hat_coeffs = fourier_coeffs(y_hat, t)

    return (time.time() - start_time) / len(xs)
def main(args):
    X_list, Y_list, scales = load_data(args.data, args.num_data, args.eta)
    t=args.truncation
    tr = args.num_trains
    cv = args.num_cvs
    cv_x = X_list[tr:tr+cv]
    cv_y = Y_list[tr:tr+cv]
    train_x = X_list[:tr]
    train_y = Y_list[:tr]
    X_test = X_list[tr+cv:]
    Y_test = Y_list[tr+cv:]
    cv_coeffs = [[fourier_coeffs(cv_x[i], t), fourier_coeffs(cv_y[i], t)] for i in range(cv)]
    training_coeffs = [[fourier_coeffs(train_x[i], t), fourier_coeffs(train_y[i], t)] for i in range(tr)]
    e = Estimator(training_coeffs, cv_coeffs, t)
    e.cross_validation()
    X_cal, Y_cal = [], []
    if args.data == "trunc_gaussians":
        X_cal, Y_cal = generate_toy_mixture_data(
            num_data=args.num_cals,
            eta=args.eta,
            seed=42
        )
    elif args.data == "mixture_of_betas":
        def T0(x): return zeta_k(np.asarray(x), 4)
        for i in range(args.num_cals):
            params = random_beta_mixture_params()
            Xi = sample_beta_mixture(1000, params)
            T_eps = random_error_map(20, k_support=[1,2,3,4])
            Yi = push_forward_samples(push_forward_samples(Xi, T0), T_eps)
            X_cal.append(Xi)
            Y_cal.append(Yi)
    cal_coeffs = [[fourier_coeffs(X_cal[j], t), fourier_coeffs(Y_cal[j], t)] for j in range(args.num_cals)]

    xs = X_test[:1000]
    ys = Y_test[:1000]

    marginal_times = []

    alpha = 0.1
    sizes = [10000, 50000, 100000, 150000, 200000]

    for i in sizes:
        # -------------------------
        # Marginal calibration
        # -------------------------
        cal_scores = []
        for j in range(i):
            xj = cal_coeffs[j][0]
            yj = cal_coeffs[j][1]

            yj_hat_coeffs = e.regress(xj)
            sj = L2_distance(yj, yj_hat_coeffs)
            cal_scores.append(sj)

        cal_scores = np.asarray(cal_scores)

        q_level = np.ceil((i + 1) * (1 - alpha)) / (i + 1)
        qhat = np.quantile(cal_scores, q_level, interpolation="higher")
        ys_hat = []
        for x in xs:
            y_hat = e.regress(fourier_coeffs(x, t), return_samples=True)
            ys_hat.append(y_hat)

        # -------------------------
        # Marginal timing
        # -------------------------
        start_time = time.time()
        for h in range(len(xs)):
            fourier_coeffs(ys_hat[h], t)
        marginal_times.append((time.time() - start_time) / len(xs))

        # -------------------------
        # Conditional timing
        # -------------------------
        Ks = [int(i/2), int(i/4), int(math.sqrt(i))]
        cal_x_coeffs = np.array(
            [c[0] for c in cal_coeffs[:i]]
        )
        conditional_times_ks = [calculate_conditional_times(cal_x_coeffs, K, xs, ys_hat, cal_scores, alpha, t) for K in Ks]
        print(f"Done size {i}, marginal time: {marginal_times[-1]}, conditional times: {conditional_times_ks}")


if __name__ == "__main__":
    args = parse_args()
    main(args)