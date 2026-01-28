import numpy as np

def load_mortality_data(file_path="mortality_data.npz"):
    data = np.load(file_path, allow_pickle=True)
    X_list, Y_list = [], []
    scales = []

    for i in range(data.shape[0]):
        X = data[i, 0].astype(np.float32)
        Y = data[i, 1].astype(np.float32)

        if len(X) == 0 or len(Y) == 0:
            continue

        mn = min(X.min(), Y.min())
        mx = max(X.max(), Y.max())

        X_list.append((X - mn) / (mx - mn + 1e-8))
        Y_list.append((Y - mn) / (mx - mn + 1e-8))
        scales.append((mn, mx))
    return X_list, Y_list, scales