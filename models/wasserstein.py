import math
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from scipy.interpolate import interp1d
def project_to_monotone(q):
    """
    Enforce non-decreasing quantile function via isotonic regression.
    """
    from sklearn.isotonic import IsotonicRegression
    x = np.arange(len(q))
    ir = IsotonicRegression(increasing=True)
    return ir.fit_transform(x, q)
class WassersteinRegression:
    def __init__(self, n_fpc_x=5, n_fpc_y=5):
        self.n_fpc_x = n_fpc_x
        self.n_fpc_y = n_fpc_y

    def fit(self, X, Y):
        """
        X, Y: shape (n_samples, n_grid)
        """
        self.X = X
        self.Y = Y
        n, m = X.shape

        # === (1) Fréchet means (mean quantiles)
        self.QX_mean = X.mean(axis=0)
        self.QY_mean = Y.mean(axis=0)

        # === (2) Log maps
        LogX = X - self.QX_mean
        LogY = Y - self.QY_mean

        # === (3) FPCA
        self.pca_X = PCA(n_components=self.n_fpc_x)
        self.pca_Y = PCA(n_components=self.n_fpc_y)

        xi = self.pca_X.fit_transform(LogX)     # (n, J)
        eta = self.pca_Y.fit_transform(LogY)    # (n, K)

        self.phi = self.pca_X.components_.T     # (m, J)
        self.psi = self.pca_Y.components_.T     # (m, K)

        # === (4) Regression: solve b_jk
        reg = LinearRegression(fit_intercept=False)
        reg.fit(xi, eta)

        self.B = reg.coef_   # shape (K, J)

        # === (5) Reconstruct beta(s,t)
        self.beta = self.phi @ self.B.T @ self.psi.T
        # beta shape: (m, m)

        return self
    def predict(self, X_new):
        """
        X_new: shape (n_new, n_grid)
        """
        # Log map
        LogX_new = X_new - self.QX_mean

        # Project to PCA space
        xi_new = self.pca_X.transform(LogX_new)

        # Predict LogY
        LogY_pred = xi_new @ self.B.T @ self.psi.T

        # Exp map
        Y_pred = self.QY_mean + LogY_pred

        # Enforce monotonicity
        Y_pred_proj = np.array([
            project_to_monotone(y) for y in Y_pred
        ])

        return Y_pred_proj

