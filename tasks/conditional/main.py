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
        default=3200,
        help="Number of data points",
    )
    parser.add_argument(
        "--eta",
        type=int,
        default=1000,
        help="Number of samples each distribution",
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
        "--num_groups",
        type=int,
        default=10,
        help="Number of groups for groupwise coverage evaluation",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ot_map",
        help="Model to use: ot_map, wasserstein, nonparametric",
    )
    parser.add_argument(
        "--file_path",
        type=str,
        default=None,
        help="File path for loading data"
    )
    opt = parser.parse_args()
    return opt
def conditional_mortality_adaptive(
    args,
    X_list,
    Y_list,
    n_repeats=20,
    alpha=0.1,
):
    """
    Adaptive CP for mortality data.
    Small-data regime:
      - repeat random splits
      - no grouping
      - report average adaptive conditional coverage
    """

    t = args.truncation
    grid = make_grid(200)

    tr = args.num_trains
    cv = args.num_cvs
    cal = args.num_cals
    K = min(args.k_neighbors, cal)  # IMPORTANT

    n = len(X_list)

    marginal_coverages = []
    adaptive_coverages = []

    for rep in range(n_repeats):
        # ----------------------
        # random split
        # ----------------------
        perm = np.random.permutation(n)

        train_idx = perm[:tr]
        cv_idx    = perm[tr:tr+cv]
        cal_idx   = perm[tr+cv:tr+cv+cal]
        test_idx  = perm[tr+cv+cal:]

        train_x = [X_list[i] for i in train_idx]
        train_y = [Y_list[i] for i in train_idx]
        cv_x    = [X_list[i] for i in cv_idx]
        cv_y    = [Y_list[i] for i in cv_idx]
        cal_x   = [X_list[i] for i in cal_idx]
        cal_y   = [Y_list[i] for i in cal_idx]
        test_x  = [X_list[i] for i in test_idx]
        test_y  = [Y_list[i] for i in test_idx]

        # ----------------------
        # model fitting
        # ----------------------
        cal_scores = []
        cal_x_coeffs = []

        if args.model == "ot_map":
            z_hat, _ = estimate_T_hat(train_x, train_y, grid)
            T_hat = piecewise_linear_interp(grid.x, z_hat)

            for x, y in zip(cal_x, cal_y):
                y_hat = T_hat(x)
                si = L2_distance(
                    fourier_coeffs(y, t),
                    fourier_coeffs(y_hat, t),
                )
                cal_scores.append(si)
                cal_x_coeffs.append(fourier_coeffs(x, t))

        elif args.model == "nonparametric":
            cv_coeffs = [[fourier_coeffs(cv_x[i], t), fourier_coeffs(cv_y[i], t)]
                         for i in range(len(cv_x))]
            train_coeffs = [[fourier_coeffs(train_x[i], t), fourier_coeffs(train_y[i], t)]
                            for i in range(len(train_x))]

            e = Estimator(train_coeffs, cv_coeffs, t)
            e.cross_validation()

            for x, y in zip(cal_x, cal_y):
                y_hat_coeffs = e.regress(fourier_coeffs(x, t))
                si = L2_distance(fourier_coeffs(y, t), y_hat_coeffs)
                cal_scores.append(si)
                cal_x_coeffs.append(fourier_coeffs(x, t))
        elif args.model == "wasserstein":
            q = np.linspace(0.01, 0.99, 500)

            # quantile representations
            X_q = [quantile_from_samples(x, q) for x in X_list]
            Y_q = [quantile_from_samples(y, q) for y in Y_list]

            wr = WassersteinRegression(n_fpc_x=5, n_fpc_y=5)
            wr.fit(
                np.asarray([X_q[i] for i in train_idx]),
                np.asarray([Y_q[i] for i in train_idx]),
            )

            for i in cal_idx:
                x = X_list[i]
                y = Y_list[i]

                y_hat = wr.predict([X_q[i]])[0]
                y_hat_coeffs = fourier_coeffs_quantile(y_hat, t)

                si = L2_distance(
                    fourier_coeffs(y, t),
                    y_hat_coeffs,
                )
                cal_scores.append(si)
                cal_x_coeffs.append(fourier_coeffs(x, t))

        cal_scores = np.asarray(cal_scores)
        cal_x_coeffs = np.asarray(cal_x_coeffs)

        # ----------------------
        # global (marginal) q_hat
        # ----------------------
        k = int(np.ceil((len(cal_scores) + 1) * (1 - alpha))) - 1
        k = max(0, min(k, len(cal_scores) - 1))

        q_hat_global = np.sort(cal_scores)[k]
        # ----------------------
        # nearest neighbors on cal set
        # ----------------------
        nn = NearestNeighbors(
            n_neighbors=K,
            metric="euclidean"
        )
        nn.fit(cal_x_coeffs)

        # ----------------------
        # test
        # ----------------------
        covered_marginal = 0
        covered_adaptive = 0

        for x, y in zip(test_x, test_y):
            x_coeffs = fourier_coeffs(x, t)

            if args.model == "ot_map":
                y_hat = T_hat(x)
                y_hat_coeffs = fourier_coeffs(y_hat, t)
            elif args.model == "wasserstein":
                y_hat = wr.predict([quantile_from_samples(x, q)])[0]
                y_hat_coeffs = fourier_coeffs_quantile(y_hat, t)
            else:
                y_hat_coeffs = e.regress(x_coeffs)

            si = L2_distance(fourier_coeffs(y, t), y_hat_coeffs)

            # marginal
            if si <= q_hat_global:
                covered_marginal += 1

            # adaptive
            _, idx = nn.kneighbors(
                x_coeffs.reshape(1, -1),
                n_neighbors=K,
                return_distance=True
            )
            local_scores = cal_scores[idx[0]]
            local_scores = np.sort(local_scores)

            k_star = int(ceil((K + 1) * (1 - alpha))) - 1
            k_star = max(0, min(k_star, K - 1))
            q_alpha = local_scores[k_star]

            if si <= q_alpha:
                covered_adaptive += 1

        marginal_coverages.append(covered_marginal / len(test_x))
        adaptive_coverages.append(covered_adaptive / len(test_x))

    print("===================================")
    print(f"Model: {args.model}")
    print(f"Repeats: {n_repeats}")
    print(f"Average marginal coverage: {np.mean(marginal_coverages):.4f}")
    print(f"Average adaptive conditional coverage: {np.mean(adaptive_coverages):.4f}")
    print("===================================")

def conditional_1d(args, X_list, Y_list):
    t=args.truncation
    grid = make_grid(200)
    tr = args.num_trains
    cv = args.num_cvs
    cal = args.num_cals
    cal_x, cal_y = X_list[tr+cv:tr+cv+cal], Y_list[tr+cv:tr+cv+cal]
    cv_x, cv_y = X_list[tr:tr+cv], Y_list[tr:tr+cv]
    train_x, train_y = X_list[:tr], Y_list[:tr]
    X_test, Y_test = X_list[tr+cv+cal:], Y_list[tr+cv+cal:]
    cal_coeffs = [[fourier_coeffs(cal_x[i], t), fourier_coeffs(cal_y[i], t)] for i in range(cal)]
    q_level = np.ceil((len(cal_coeffs)+1)*(1-0.1))/len(cal_coeffs)
    cal_scores = []
    q_hat = None
    cal_x_coeffs = np.array([c[0] for c in cal_coeffs])

    nn = NearestNeighbors(
        n_neighbors=args.k_neighbors,
        metric='euclidean'
    )
    nn.fit(cal_x_coeffs)
    if args.model == "ot_map":
        z_hat, _ = estimate_T_hat(train_x, train_y, grid) # isotonic regression
        T_hat = piecewise_linear_interp(grid.x, z_hat) # estimate model isotonic
        for i in range(len(cal_x)):
            x_cal = cal_x[i]
            y_cal = cal_coeffs[i][1]
            y_cal_hat = T_hat(x_cal)
            y_hat_coeffs = fourier_coeffs(y_cal_hat, t)
            si = L2_distance(y_cal, y_hat_coeffs)
            cal_scores.append(si)
    elif args.model == "nonparametric":
        cv_coeffs = [[fourier_coeffs(cv_x[i], t), fourier_coeffs(cv_y[i], t)] for i in range(cv)]
        training_coeffs = [[fourier_coeffs(train_x[i], t), fourier_coeffs(train_y[i], t)] for i in range(tr)]
        e = Estimator(training_coeffs, cv_coeffs, t) # nonparametric
        e.cross_validation()
        for i in range(len(cal_x)):
            x_cal = cal_x[i]
            y_cal = cal_coeffs[i][1]
            y_hat_coeffs = e.regress(fourier_coeffs(x_cal, t))
            si = L2_distance(y_cal, y_hat_coeffs)
            cal_scores.append(si)
    elif args.model == "wasserstein":
        q = np.linspace(0.01, 0.99, 1000)
        X_list_q = [quantile_from_samples(x, q) for x in X_list]
        Y_list_q = [quantile_from_samples(y, q) for y in Y_list]
        X_train_w = np.asarray(X_list_q[:tr])
        Y_train_w = np.asarray(Y_list_q[:tr])
        X_cal_w = np.asarray(X_list_q[tr+cv:tr+cv+cal])
        X_test_w = np.asarray(X_list_q[tr+cv+cal:])
        wr = WassersteinRegression(n_fpc_x=5, n_fpc_y=5)
        wr.fit(X_train_w, Y_train_w)
        for i in range(len(X_cal_w)):
            x_cal = X_cal_w[i]
            y_cal = cal_coeffs[i][1]
            y_cal_hat = wr.predict([x_cal])[0]
            y_hat_coeffs = fourier_coeffs_quantile(y_cal_hat, t)
            si = L2_distance(y_cal, y_hat_coeffs)
            cal_scores.append(si)
    q_hat = np.quantile(cal_scores, q_level, interpolation='higher')
    covered = 0
    for i in range(len(X_test)):
        xi = X_test[i]
        yi = Y_test[i]
        yi_coeffs = fourier_coeffs(yi, t)
        yi_hat_coeffs = None
        if args.model == "ot_map":
            y_hat = T_hat(xi)
            yi_hat_coeffs = fourier_coeffs(y_hat, t)
        elif args.model == "wasserstein":
            x_w = X_test_w[i]
            y_hat = wr.predict([x_w])[0]
            yi_hat_coeffs = fourier_coeffs_quantile(y_hat, t)
        elif args.model == "nonparametric":
            yi_hat_coeffs = e.regress(fourier_coeffs(xi, t))
        si = L2_distance(yi_coeffs, yi_hat_coeffs)
        if si <= q_hat:
            covered += 1
    marginal_coverage = covered/len(X_test)
    print(f"Marginal coverages: {marginal_coverage}") 
    K = args.k_neighbors
    cal_scores = np.asarray(cal_scores)
    prediction_set_size = []
    for i in range(len(X_test)):
        x = X_test[i]
        y = Y_test[i]
        x_coeffs = np.asarray(fourier_coeffs(x, t)).reshape(1, -1)
        _, indices = nn.kneighbors(x_coeffs, n_neighbors=K, return_distance=True)
        neighbor_idx = indices[0]
        local_scores = cal_scores[neighbor_idx]
        sorted_scores = np.sort(local_scores)

        k_star = int(ceil((K + 1) * (1 - 0.1))) - 1
        k_star = max(0, min(k_star, K - 1))
        q_alpha = sorted_scores[k_star]
        prediction_set_size.append(q_alpha)
    prediction_set_size = np.asarray(prediction_set_size)
    n = args.num_groups
    sorted_idx = np.argsort(prediction_set_size)
    groups_idx = np.array_split(sorted_idx, n)

    groups = []
    for g in groups_idx:
        groups.append({
            "test_indices": g,
            "prediction_set_sizes": prediction_set_size[g]
        })
    coverages = []
    for g in groups:
        idx = g["test_indices"]
        qhat = g["prediction_set_sizes"]

        covered = 0
        for k, j in enumerate(idx):
            x = X_test[j]
            y = Y_test[j]
            if args.model == "ot_map":
                y_hat = T_hat(x)
                y_hat_coeffs = fourier_coeffs(y_hat, t)
            elif args.model == "wasserstein":
                x_w = X_test_w[j]
                y_hat = wr.predict([x_w])[0]
                y_hat_coeffs = fourier_coeffs_quantile(y_hat, t)
            elif args.model == "nonparametric":
                y_hat_coeffs = e.regress(fourier_coeffs(x, t))
            y_coeffs = fourier_coeffs(y, t)

            sj = L2_distance(y_coeffs, y_hat_coeffs)

            if sj <= qhat[k]:
                covered += 1

        coverages.append(covered / len(idx))
        print(f"Group coverage: {covered/len(idx)}, group interval: {np.min(qhat)} - {np.max(qhat)}")
def conditional_nd(args, X_list, Y_list):
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
    alpha = 0.1

    cal_scores = []
    for i in range(len(cal_coeffs_in)):
        y_coeffs = E.full_regress(cal_coeffs_in[i])
        si = L2_distance(cal_coeffs_out[i], y_coeffs)
        cal_scores.append(si)
    n = len(cal_scores)
    k = int(np.ceil((n + 1) * (1 - alpha))) - 1
    k = min(max(k, 0), n - 1)

    q_hat = np.sort(cal_scores)[k]
    covered = 0
    for i in range(len(test_coeffs_in)):
        y_coeffs = test_hat_coeffs[i]
        si = L2_distance(test_coeffs_out[i], y_coeffs)
        if si <= q_hat:
            covered += 1
    print(covered/len(test_coeffs_in))
    nn = NearestNeighbors(
        n_neighbors=args.k_neighbors,
        metric='euclidean'
    )
    nn.fit(cal_coeffs_in)
    prediction_set_size = []
    K = args.k_neighbors
    cal_scores = np.asarray(cal_scores)
    for i in range(len(test_coeffs_in)):
        x = test_coeffs_in[i]
        y = test_coeffs_out[i]
        y_hat_coeffs = test_hat_coeffs[i]
        x_coeffs = np.asarray(x).reshape(1, -1)
        distances, indices = nn.kneighbors(x_coeffs, n_neighbors=K, return_distance=True)
        neighbor_idx = indices[0]

        local_scores = cal_scores[neighbor_idx]
        sorted_scores = np.sort(local_scores)
        k_star = int(ceil((K + 1) * (1 - 0.1))) - 1
        k_star = max(0, min(k_star, K - 1))
        q_alpha = sorted_scores[k_star]
        prediction_set_size.append(q_alpha)
    prediction_set_size = np.asarray(prediction_set_size)

    n = args.num_groups


    sorted_idx = np.argsort(prediction_set_size)
    sorted_sizes = prediction_set_size[sorted_idx]

    groups_idx = np.array_split(sorted_idx, n)

    groups = []
    for g in groups_idx:
        groups.append({
            "test_indices": g,
            "prediction_set_sizes": prediction_set_size[g]
        })
    coverages = []

    for g in groups:
        idx = g["test_indices"]
        qhat = g["prediction_set_sizes"]

        covered = 0
        for k, j in enumerate(idx):
            x = test_coeffs_in[j]
            y = test_coeffs_out[j]
            y_hat_coeffs = test_hat_coeffs[j]

            sj = L2_distance(y, y_hat_coeffs)

            if sj <= qhat[k]:
                covered += 1
        print(f"Group coverage: {covered/len(idx)}, group interval: {np.min(qhat)} - {np.max(qhat)}")
        coverages.append(covered / len(idx))
    print(coverages)

def main(args):
    X_list, Y_list, _ = load_data(args.data, args.num_data, args.eta, args.dim, args.file_path)
    if args.dim == 1:
        if args.data == "mortality":
            conditional_mortality_adaptive(args, X_list, Y_list)
        else:
            conditional_1d(args, X_list, Y_list)
    else:
        conditional_nd(args, X_list, Y_list)
if __name__ == "__main__":
    args = parse_args()
    main(args)