import argparse
from math import ceil

from matplotlib import pyplot as plt
import numpy as np

from models.nonparametric import Estimator
from models.ot_map import estimate_T_hat, piecewise_linear_interp
from models.wasserstein import WassersteinRegression
from utils.helpers import L2_distance, fourier_coeffs, fourier_coeffs_quantile, load_data, make_grid, quantile_from_samples, quantile_from_samples
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
    opt = parser.parse_args()
    return opt
def main(args):
    X_list, Y_list, _ = load_data(args.data, args.num_data, args.eta)
    models = []
    ts = range(2,21, 1)
    grid = make_grid(200)
    tr = args.num_trains
    cv = args.num_cvs
    cal = args.num_cals
    cal_x = X_list[tr:tr+cal]
    cal_y = Y_list[tr:tr+cal]
    cv_x = X_list[tr+cal:tr+cal+cv]
    cv_y = Y_list[tr+cal:tr+cal+cv]
    train_x = X_list[:tr]
    train_y = Y_list[:tr]
    z_hat, info = estimate_T_hat(train_x, train_y, grid)
    T_hat = piecewise_linear_interp(grid.x, z_hat) 
    for t in ts:
        cal_coeffs = [[fourier_coeffs(cal_x[i], t), fourier_coeffs(cal_y[i], t)] for i in range(cal)]
        cv_coeffs = [[fourier_coeffs(cv_x[i], t), fourier_coeffs(cv_y[i], t)] for i in range(cv)]
        training_coeffs = [[fourier_coeffs(train_x[i], t), fourier_coeffs(train_y[i], t)] for i in range(tr)]
        e = Estimator(training_coeffs, cv_coeffs, t) 
        e.cross_validation()
        models.append(e)
    q = np.linspace(0.01, 0.99, 1000)
    X_list_q = [quantile_from_samples(x, q) for x in X_list]
    Y_list_q = [quantile_from_samples(y, q) for y in Y_list]
    X_train_w = X_list_q[:tr]
    Y_train_w = Y_list_q[:tr]
    X_cal_w = X_list_q[tr+cv:tr+cv+cal]
    Y_cal_w = Y_list_q[tr+cv:tr+cv+cal]
    X_train_w = np.asarray(X_train_w)
    Y_train_w = np.asarray(Y_train_w)
    X_cal_w = np.asarray(X_cal_w)
    Y_cal_w = np.asarray(Y_cal_w)
    wr = WassersteinRegression(n_fpc_x=5, n_fpc_y=5)
    wr.fit(X_train_w, Y_train_w)
    alpha = 0.1
    q_level = np.ceil((len(cal_coeffs)+1)*(1-alpha))/len(cal_coeffs)
    qhats_1 = []
    qhats_2 = []
    qhats_3 = []
    j = 0
    for t in ts:
        cal_coeffs = [[fourier_coeffs(cal_x[i], t), fourier_coeffs(cal_y[i], t)] for i in range(1000)]
        cal_coeffs_w = [[fourier_coeffs_quantile(X_cal_w[i], t), fourier_coeffs_quantile(Y_cal_w[i], t)] for i in range(1000)]
        cal_scores_1 = []
        cal_scores_2 = []
        cal_scores_3 = []
        for i in range(len(cal_coeffs)):
            x_cal = cal_coeffs[i][0]
            y_cal = cal_coeffs[i][1]
            x_cal_w = X_cal_w[i]
            y_cal_w = cal_coeffs_w[i][1]
            y_cal_hat1 = T_hat(cal_x[i])
            y_cal_hat2 = models[j].regress(x_cal)
            y_cal_hat3 = wr.predict([x_cal_w])[0]
            y_cal_hat3_coeffs = fourier_coeffs_quantile(y_cal_hat3, t)
            y_cal_hat1_coeffs = fourier_coeffs(y_cal_hat1, t)
            si1 = L2_distance(y_cal, y_cal_hat1_coeffs)
            si2 = L2_distance(y_cal, y_cal_hat2)
            si3 = L2_distance(y_cal_w, y_cal_hat3_coeffs)
            cal_scores_1.append(si1)
            cal_scores_2.append(si2)
            cal_scores_3.append(si3)
        j += 1
        print(f"Done {t}")
        qhats_1.append(np.quantile(cal_scores_1, q_level, interpolation='higher'))
        qhats_2.append(np.quantile(cal_scores_2, q_level, interpolation='higher'))
        qhats_3.append(np.quantile(cal_scores_3, q_level, interpolation='higher'))
    print("OT map regression qhats:", qhats_1)
    print("Nonparametric regression qhats:", qhats_2)
    print("Wasserstein regression qhats:", qhats_3)
    
if __name__ == "__main__":
    args = parse_args()
    main(args)