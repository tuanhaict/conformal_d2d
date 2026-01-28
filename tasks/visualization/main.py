import argparse
from math import ceil

from matplotlib import pyplot as plt
import numpy as np
from models.nonparametric import Estimator
from utils.helpers import L2_distance, coeffs_to_approx_density, fourier_coeffs, load_data, sample_ball_uniform_Lp
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
        "--file_path",
        type=str,
        default=None,
        help="File path for loading data"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed"
    )
    opt = parser.parse_args()
    return opt
def main(args):
    np.random.seed(args.seed)
    X_list, Y_list, scales = load_data(args.data, args.num_data, args.eta, file_path=args.file_path)
    perm = np.random.permutation(len(X_list))
    X_list = [X_list[i] for i in perm]
    Y_list = [Y_list[i] for i in perm]
    scales = [scales[i] for i in perm] if len(scales) != 0 else scales
    t=args.truncation
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
    cal_coeffs = [[fourier_coeffs(cal_x[i], t), fourier_coeffs(cal_y[i], t)] for i in range(cal)]
    cv_coeffs = [[fourier_coeffs(cv_x[i], t), fourier_coeffs(cv_y[i], t)] for i in range(cv)]
    training_coeffs = [[fourier_coeffs(train_x[i], t), fourier_coeffs(train_y[i], t)] for i in range(tr)]
    e = Estimator(training_coeffs, cv_coeffs, t)
    e.cross_validation()
    cal_scores = []
    for i in range(len(cal_x)):
        x_cal = cal_x[i]
        y_cal = cal_coeffs[i][1]
        y_hat_coeffs = e.regress(fourier_coeffs(x_cal, t))
        si = L2_distance(y_cal, y_hat_coeffs)
        cal_scores.append(si)
    q_level = np.ceil((len(cal_coeffs)+1)*(1-0.1))/len(cal_coeffs)
    q_hat = np.quantile(cal_scores, q_level, interpolation='higher')
    i = np.random.randint(0, len(X_test))
    xi = X_test[i]
    yi = Y_test[i]
    yi_coeffs = fourier_coeffs(yi, t)
    yi_hat_coeffs = e.regress(fourier_coeffs(xi, t))
    cxi = sample_ball_uniform_Lp(yi_hat_coeffs, q_hat, 100, p=2)
    yi_pdf = coeffs_to_approx_density(yi_coeffs)
    x = np.linspace(0, 1, 500)
    yi_true = np.array([yi_pdf(val) for val in x])
    plt.figure(figsize=(8,6))
    if len(scales) != 0:
        x_real = scales[0][0] + x * (scales[0][1] - scales[0][0])
    else:
        x_real = x
    for i in range(len(cxi)):
        yk = cxi[i]
        yk_pdf = coeffs_to_approx_density(yk)
        yk_true = np.array([yk_pdf(val) for val in x])
        plt.plot(x_real, yk_true, color="blue", alpha=0.05)

    plt.plot(x_real, yi_true, color="red", linewidth=2.5, label="True density")
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel("x", fontsize=16)
    plt.ylabel("density", fontsize=16)
    plt.legend(loc='upper right', fontsize=16, frameon=False)
    plt.grid(alpha=0.2)
    plt.savefig(f"{args.data}.pdf", format="pdf", bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    args = parse_args()
    main(args)