import argparse
from math import ceil
import matplotlib.pyplot as plt
from scipy.stats import betabinom
import numpy as np
from sklearn.neighbors import NearestNeighbors

from models.nonparametric import Estimator, ND_Estimator
from models.ot_map import estimate_T_hat, piecewise_linear_interp
from models.wasserstein import WassersteinRegression
from utils.helpers import L2_distance, fourier_coeffs, fourier_coeffs_ND_fast, fourier_coeffs_quantile, load_data, make_grid, quantile_from_samples, quantile_from_samples
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
        default=6200,
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
        "--dim",
        type=int,
        default=1,
        help="Dimension of the data",
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
        default=1000,
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
        help="Number of neighbors for adaptive CP",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ot_map",
        help="Model type: ot_map, nonparametric, wasserstein_regression",
    )
    parser.add_argument(
        "--num_tests",
        type=int,
        default=1000,
        help="Number of test samples",
    )
    parser.add_argument(
        "--num_iter",
        type=int,
        default=50000,
        help="Number of iterations for calculating empirical coverage",
    )
    parser.add_argument(
        "--file_path",
        type=str,
        default=None,
        help="File path for loading data"
    )
    opt = parser.parse_args()
    return opt
def plot_empirical_coverage(coverages, num_cals, num_tests):
    n = num_cals              # calibration size
    n_test = num_tests         # number of test points
    alpha = 0.1
    bin_width = 0.001
    xmin, xmax = 0.82, 0.98

    # --- Beta parameters from theorem ---
    l = int(np.floor((n + 1) * alpha))
    a_param = n + 1 - l
    b_param = l

    coverages_array = np.array(coverages)

    # --- histogram of empirical coverages (density) ---
    bins = np.arange(xmin, xmax + bin_width, bin_width)
    hist_density, bin_edges = np.histogram(
        coverages_array, bins=bins, density=True
    )
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # --- Beta–Binomial distribution ---
    k_min = int(np.ceil(xmin * n_test))
    k_max = int(np.floor(xmax * n_test))
    k = np.arange(k_min, k_max + 1)

    # pmf of k = number of covered points
    pmf = betabinom.pmf(k, n_test, a_param, b_param)

    # convert to density on coverage axis
    coverage_vals = k / n_test
    bb_density = pmf * n_test   # Jacobian: dk = n_test * dC

    # --- plotting ---
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.bar(
        bin_centers, hist_density,
        width=bin_width, alpha=0.6,
        label='Empirical', edgecolor='none'
    )

    ax.plot(
        coverage_vals, bb_density,
        'r-', lw=2.5,
        label=rf'Beta-Binomial$(n_{{test}}={n_test})$'
    )

    ax.set_xlim(xmin, xmax)
    ax.set_xlabel('Coverage', fontsize=16)
    ax.set_ylabel('Density', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax.legend(loc='upper left', fontsize=14, framealpha=0.0)
    ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig("beta_binomial.pdf", format="pdf", bbox_inches="tight")
    plt.show()
def empirical_coverages(data_in_coeffs, data_out_coeffs, data_hat_coeffs, num_cals =1000, num_tests=1000):
    coverages = []
    K = num_cals
    alpha = 0.1
    for j in range(args.num_iter):
        perm = np.random.permutation(len(data_in_coeffs))
        data_in_coeffs = [data_in_coeffs[i] for i in perm]
        data_out_coeffs = [data_out_coeffs[i] for i in perm]
        data_hat_coeffs = [data_hat_coeffs[i] for i in perm]
        cal_scores = []
        for i in range(num_cals):
            yi_hat_coeffs = data_hat_coeffs[i]
            yi_coeffs = data_out_coeffs[i]
            si = L2_distance(yi_coeffs, yi_hat_coeffs)
            cal_scores.append(si)
        k = int(np.ceil((K + 1) * (1 - alpha))) - 1
        k = min(max(k, 0), K - 1)
        q_hat = np.sort(cal_scores)[k]
        covered=0
        for i in range(num_tests):
            yi_hat_coeffs = data_hat_coeffs[num_cals+i]
            yi_coeffs = data_out_coeffs[num_cals+i]
            si = L2_distance(yi_coeffs, yi_hat_coeffs)
            if si <= q_hat:
                covered +=1
        coverages.append(covered/num_tests)
        if j % 1000 == 0:
            print(f"{j} coverage: {coverages[-1]}")
    return coverages
def beta_binomial_1d(args, X_list, Y_list):
    model = args.model
    t=args.truncation
    grid = make_grid(200)
    tr = args.num_trains
    cv = args.num_cvs
    cv_x = X_list[tr:tr+cv]
    cv_y = Y_list[tr:tr+cv]
    train_x = X_list[:tr]
    train_y = Y_list[:tr]
    data_in = X_list[tr+cv:]
    data_out = Y_list[tr+cv:]
    data_in_coeffs = [fourier_coeffs(x, t) for x in data_in]
    data_out_coeffs = [fourier_coeffs(y, t) for y in data_out]
    data_hat_coeffs = None
    if model == "ot_map":
        z_hat, info = estimate_T_hat(train_x, train_y, grid) 
        T_hat = piecewise_linear_interp(grid.x, z_hat)
        data_hat_coeffs = [fourier_coeffs(T_hat(x), t) for x in data_in]
    elif model == "nonparametric":
        cv_coeffs = [[fourier_coeffs(cv_x[i], t), fourier_coeffs(cv_y[i], t)] for i in range(cv)]
        training_coeffs = [[fourier_coeffs(train_x[i], t), fourier_coeffs(train_y[i], t)] for i in range(tr)]
        e = Estimator(training_coeffs, cv_coeffs, t)
        e.cross_validation()
        data_hat_coeffs = [e.regress(x) for x in data_in_coeffs]
    else:
        q = np.linspace(0.01, 0.99, 1000)
        X_list_q = [quantile_from_samples(x, q) for x in X_list]
        Y_list_q = [quantile_from_samples(y, q) for y in Y_list]
        X_train_w = X_list_q[:tr]
        Y_train_w = Y_list_q[:tr]
        data_in_w = X_list_q[tr+cv:]
        X_train_w = np.asarray(X_train_w)
        Y_train_w = np.asarray(Y_train_w)
        wr = WassersteinRegression(n_fpc_x=5, n_fpc_y=5)
        wr.fit(X_train_w, Y_train_w)
        data_hat_coeffs = [fourier_coeffs_quantile(wr.predict([x])[0], t) for x in data_in_w]
    coverages = empirical_coverages(data_in_coeffs, data_out_coeffs, data_hat_coeffs, num_cals = args.num_cals, num_tests = args.num_tests)
    plot_empirical_coverage(coverages, args.num_cals, args.num_tests)

def beta_binomial_nd(args, X_list, Y_list):
    T = args.truncation
    D = args.dim
    dists_coeffs_in = [fourier_coeffs_ND_fast(x, T, D) for x in X_list]
    dists_coeffs_out = [fourier_coeffs_ND_fast(x, T, D) for x in Y_list]
    tr = args.num_trains
    cv = args.num_cvs
    cal = args.num_cals
    train_coeffs_in = dists_coeffs_in[:tr]
    train_coeffs_out = dists_coeffs_out[:tr]
    cv_coeffs_in = dists_coeffs_in[tr:tr+cv]
    cv_coeffs_out = dists_coeffs_out[tr:tr+cv]
    cal_coeffs_in = dists_coeffs_in[tr+cv:tr+cv+cal]
    cal_coeffs_out = dists_coeffs_out[tr+cv:tr+cv+cal]
    test_coeffs_in  = dists_coeffs_in[tr+cv+cal:]
    test_coeffs_out = dists_coeffs_out[tr+cv+cal:]
    E = ND_Estimator(train_coeffs_in, train_coeffs_out, cv_coeffs_in, cv_coeffs_out, degree = T, dim = D)
    E.train()
    test_hat_coeffs = [E.full_regress(x) for x in test_coeffs_in]
    cal_hat_coeffs = [E.full_regress(x) for x in cal_coeffs_in]
    data_hat_coeffs = test_hat_coeffs + cal_hat_coeffs
    data_coeffs_in = test_coeffs_in + cal_coeffs_in
    data_coeffs_out = test_coeffs_out + cal_coeffs_out
    coverages = []
    K = args.num_cals
    alpha = 0.1

    for i in range(args.num_iter):
        perm = np.random.permutation(len(data_coeffs_in))
        cal_scores = [L2_distance(data_coeffs_out[j], data_hat_coeffs[j]) for j in perm[:K]]
        k = int(np.ceil((K + 1) * (1 - alpha))) - 1
        k = min(max(k, 0), K - 1)
        q_hat = np.sort(cal_scores)[k]
        covered = 0
        for j in perm[K: 2*K]:
            sj = L2_distance(data_coeffs_out[j], data_hat_coeffs[j])
            if sj <= q_hat:
                covered += 1
        coverages.append(covered / K)
        if i % 1000 == 0:
            print(f"Iteration {i} coverage: {coverages[-1]}")
    plot_empirical_coverage(coverages, args.num_cals, args.num_tests)
def main(args):
    X_list, Y_list, _ = load_data(args.data, args.num_data, args.eta, args.dim, args.file_path)
    if args.dim == 1:
        beta_binomial_1d(args, X_list, Y_list)
    else:
        beta_binomial_nd(args, X_list, Y_list)
if __name__ == "__main__":
    args = parse_args()
    main(args)