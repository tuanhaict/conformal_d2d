import argparse
from math import ceil

from matplotlib import pyplot as plt
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
        "--file_path",
        type=str,
        default=None,
        help="File path for loading data"
    )
    opt = parser.parse_args()
    return opt
def plot_evc(alphas, qhats_1, qhats_2, model, data):
    colors = {
        "ot": ["#1f77b4", "#2ca02c"],
        "nonparametric": ["#ff7f0e", "#d62728"],
        "wasserstein": ["#9467bd", "#7f7f7f"]
    }
    plt.figure(figsize=(8, 6))

    plt.plot(
        1 - alphas, qhats_1,
        color=colors[model][0], linewidth=2.5,
        label=f"{model.capitalize()} regression"
    )

    plt.plot(
        1 - alphas, qhats_2,
        color=colors[model][1], linewidth=2.5,
        label=f"{model.capitalize()} regression (Cond.)"
    )
    plt.xlabel("Coverage", fontsize=16)
    plt.ylabel("Efficiency", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=15, frameon=False, loc="upper left")
    plt.tight_layout()
    plt.savefig(f"{model}_{data}_evc.pdf", format="pdf", bbox_inches="tight")
    plt.show()

def calculate_conditional_qhats(cal_coeffs, X_test, Y_test, k=500, cal_scores_1=None, cal_scores_2=None, alphas=None, cal_scores_3=None, t=10):
    cal_x_coeffs = np.array([c[0] for c in cal_coeffs])

    nn = NearestNeighbors(
        n_neighbors=k,
        metric='euclidean'
    )
    nn.fit(cal_x_coeffs)
    qhats_4 = []
    qhats_5 = []
    qhats_6 = []
    K = k
    cal_scores_1_np = np.asarray(cal_scores_1)
    cal_scores_2_np = np.asarray(cal_scores_2)
    cal_scores_3_np = np.asarray(cal_scores_3)
    for i in range(len(alphas)):
        q_list_1 = []
        q_list_2 = []
        q_list_3 = []
        for j in range(len(X_test)): # Use X_test length as the iteration limit
            x = X_test[j]
            y = Y_test[j]
            x_coeffs = np.asarray(fourier_coeffs(x, t)).reshape(1, -1)
            distances, indices = nn.kneighbors(x_coeffs, n_neighbors=K, return_distance=True)
            neighbor_idx = indices[0]
            # Use the NumPy arrays for indexing
            local_scores_1 = cal_scores_1_np[neighbor_idx]
            local_scores_2 = cal_scores_2_np[neighbor_idx]
            local_scores_3 = cal_scores_3_np[neighbor_idx]
            sorted_scores_1 = np.sort(local_scores_1)
            sorted_scores_2 = np.sort(local_scores_2)
            sorted_scores_3 = np.sort(local_scores_3)
            k_star = int(np.ceil((K + 1) * (1 - alphas[i]))) - 1
            k_star = max(0, min(k_star, K - 1))
            q_alpha_1 = sorted_scores_1[k_star]
            q_alpha_2 = sorted_scores_2[k_star]
            q_alpha_3 = sorted_scores_3[k_star]
            q_list_1.append(q_alpha_1)
            q_list_2.append(q_alpha_2)
            q_list_3.append(q_alpha_3)
        print("Finished alpha ", alphas[i])
        qhats_4.append(np.mean(q_list_1))
        qhats_5.append(np.mean(q_list_2))
        qhats_6.append(np.mean(q_list_3))
    return qhats_4, qhats_5, qhats_6
def calculate_qhats(cal_coeffs, alphas, T_hat, e, wr, cal_x, X_cal_w, t):
    alphas = np.linspace(0.04, 0.3, 100)
    cal_scores_1 = []
    cal_scores_2 = []
    cal_scores_3 = []
    for i in range(len(cal_coeffs)):
        x_cal = cal_coeffs[i][0]
        y_cal = cal_coeffs[i][1]
        y_cal_hat1 = T_hat(cal_x[i])
        y_cal_hat2_coeffs = e.regress(x_cal)
        y_cal_hat3 = wr.predict([X_cal_w[i]])[0]
        y_cal_hat1_coeffs = fourier_coeffs(y_cal_hat1, t)
        y_cal_hat3_coeffs = fourier_coeffs_quantile(y_cal_hat3, t)
        si1 = L2_distance(y_cal, y_cal_hat1_coeffs)
        si2 = L2_distance(y_cal, y_cal_hat2_coeffs)
        si3 = L2_distance(y_cal, y_cal_hat3_coeffs)
        cal_scores_1.append(si1)
        cal_scores_2.append(si2)
        cal_scores_3.append(si3)
    qhats_1 = []
    qhats_2 = []
    qhats_3 = []
    n = len(cal_coeffs)
    for i in range(len(alphas)):
        k = int(np.ceil((n + 1) * (1 - alphas[i]))) - 1
        k = min(max(k, 0), n - 1)
        q_hat_1 = np.sort(cal_scores_1)[k]
        q_hat_2 = np.sort(cal_scores_2)[k]
        q_hat_3 = np.sort(cal_scores_3)[k]
        qhats_1.append(q_hat_1)
        qhats_2.append(q_hat_2)
        qhats_3.append(q_hat_3)
    return qhats_1, qhats_2, qhats_3, cal_scores_1, cal_scores_2, cal_scores_3
def evc_1d(args, X_list, Y_list):
    print("Running EVC task in 1D")
    t=args.truncation
    grid = make_grid(200)
    tr = args.num_trains
    cv = args.num_cvs
    cal = args.num_cals
    cal_x = X_list[tr+cv:tr+cv+cal]
    cal_y = Y_list[tr+cv:tr+cv+cal]
    cv_x = X_list[tr:tr+cv]
    cv_y = Y_list[tr:tr+cv]
    train_x = X_list[:tr]
    train_y = Y_list[:tr]
    X_test = X_list[tr+cv+cal:]
    Y_test = Y_list[tr+cv+cal:]
    z_hat, info = estimate_T_hat(train_x, train_y, grid) # isotonic regression
    T_hat = piecewise_linear_interp(grid.x, z_hat) # estimate model isotonic
    cal_coeffs = [[fourier_coeffs(cal_x[i], t), fourier_coeffs(cal_y[i], t)] for i in range(cal)]
    cv_coeffs = [[fourier_coeffs(cv_x[i], t), fourier_coeffs(cv_y[i], t)] for i in range(cv)]
    training_coeffs = [[fourier_coeffs(train_x[i], t), fourier_coeffs(train_y[i], t)] for i in range(tr)]
    e = Estimator(training_coeffs, cv_coeffs, t) # nonparametric
    e.cross_validation()
    q = np.linspace(0.01, 0.99, 1000)
    X_list_q = [quantile_from_samples(x, q) for x in X_list]
    Y_list_q = [quantile_from_samples(y, q) for y in Y_list]
    X_train_w = np.asarray(X_list_q[:tr])
    Y_train_w = np.asarray(Y_list_q[:tr])
    X_cal_w = np.asarray(X_list_q[tr+cv:tr+cv+cal])
    wr = WassersteinRegression(n_fpc_x=5, n_fpc_y=5)
    wr.fit(X_train_w, Y_train_w)
    qhats_1, qhats_2, qhats_3, cal_scores_1, cal_scores_2, cal_scores_3 = calculate_qhats(cal_coeffs, np.linspace(0.04, 0.3, 100), T_hat, e, wr, cal_x, X_cal_w, t=t)
    qhats_4, qhats_5, qhats_6 = calculate_conditional_qhats(
        cal_coeffs,
        X_test,
        Y_test,
        k=args.k_neighbors,
        cal_scores_1=cal_scores_1,
        cal_scores_2=cal_scores_2,
        cal_scores_3=cal_scores_3,
        alphas=np.linspace(0.04, 0.3, 100),
        t=t
    )
    #Plot for nonparametric
    plot_evc(
        np.linspace(0.04, 0.3, 100),
        qhats_3,
        qhats_5,
        model="nonparametric",
        data=args.data
    )
    #Plot for OT map
    # plot_evc(
    #     np.linspace(0.04, 0.3, 100),
    #     qhats_1,
    #     qhats_4,
    #     model="ot_map",
    #     data=args.data
    # )
    #Plot for Wasserstein
    # plot_evc(
    #     np.linspace(0.04, 0.3, 100),
    #     qhats_3,
    #     qhats_6,
    #     model="wasserstein",
    #     data=args.data
    # )
def evc_nd(args, X_list, Y_list):
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
    cal_scores = []
    for i in range(len(cal_coeffs_in)):
        y_coeffs = E.full_regress(cal_coeffs_in[i])
        si = L2_distance(cal_coeffs_out[i], y_coeffs)
        cal_scores.append(si)
    nn = NearestNeighbors(
        n_neighbors=args.k_neighbors,
        metric='euclidean'
    )
    nn.fit(cal_coeffs_in)
    alphas = np.linspace(0.04, 0.30, 100)
    qhats_1 = []
    for i in range(len(alphas)):
        q_level = np.ceil((len(cal_coeffs_in)+1)*(1-alphas[i]))/len(cal_coeffs_in)
        qhats_1.append(np.quantile(cal_scores, q_level, interpolation='higher'))
    qhats_2 = []
    cal_scores_np = np.asarray(cal_scores)
    K = args.k_neighbors
    for i in range(len(alphas)):
        q_list = []
        for j in range(len(test_coeffs_in)):
            x = test_coeffs_in[j]
            y = test_coeffs_out[j]
            x_coeffs = np.asarray(x).reshape(1, -1)
            distances, indices = nn.kneighbors(x_coeffs, n_neighbors=K, return_distance=True)
            neighbor_idx = indices[0]
            local_scores = cal_scores_np[neighbor_idx]
            sorted_scores = np.sort(local_scores)
            k_star = int(ceil((K + 1) * (1 - alphas[i]))) - 1
            k_star = max(0, min(k_star, K - 1))
            q_alpha = sorted_scores[k_star]
            q_list.append(q_alpha)
        print(f"Finish {i}")
        qhats_2.append(np.mean(q_list))
    plot_evc(
        alphas,
        qhats_1,
        qhats_2,
        model="nonparametric",
        data=args.data
    )
def main(args):
    X_list, Y_list, _ = load_data(args.data, args.num_data, args.eta, args.dim, args.file_path)
    if args.dim == 1:
        evc_1d(args, X_list, Y_list)
    elif args.dim >=3:
        evc_nd(args, X_list, Y_list)
if __name__ == "__main__":
    args = parse_args()
    main(args)