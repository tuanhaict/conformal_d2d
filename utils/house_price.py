import numpy as np

def load_house_price_data(file_path="house_price_data.npz"):
    data = np.load(file_path, allow_pickle=True)
    X_list_raw = [data[i,0].copy() for i in range(data.shape[0])]
    Y_list_raw = [data[i, 1].copy() for i in range(data.shape[0])]

    X_list = []
    Y_list = []
    scales = []
    eps = 1e-12

    for X, Y in zip(X_list_raw, Y_list_raw):
        pair_min = min(X.min(), Y.min())
        pair_max = max(X.max(), Y.max())

        scale = pair_max - pair_min
        if scale < eps:
            Xn = np.zeros_like(X)
            Yn = np.zeros_like(Y)
        else:
            Xn = (X - pair_min) / scale
            Yn = (Y - pair_min) / scale
        scales.append((pair_min, pair_max))
        X_list.append(Xn)
        Y_list.append(Yn)
    return X_list, Y_list, scales