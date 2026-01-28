import numpy as np
from sklearn import neighbors
from utils.helpers import L2_distance, RBF_kernel, coeffs_to_approx_density, rejection_sample


class Estimator:
  def __init__(self, training_coeffs, cv_coeffs, t = 20, nu = 1., sigma = 1., dist_fn = L2_distance, kernel = RBF_kernel, bandwidths = [.15, .25]):
    self.t = t
    self.nu = nu
    self.sigma = sigma
    self.dist_fn = dist_fn
    self.kernel = kernel
    self.bandwidths = bandwidths
    self.best_b = None
    self.training_coeffs = training_coeffs
    self.cv_coeffs = cv_coeffs
  def cross_validation(self):
    print(' >>> [debug] cross-validating bandwidths...')
    b_errs = []
    for b in self.bandwidths:
      net_err = 0.
      for i in range(len(self.cv_coeffs)):
        input_coeffs = self.cv_coeffs[i][0]
        target_coeffs = self.cv_coeffs[i][1]
        Y0_coeffs = self.regress(input_coeffs, b)
        net_err += L2_distance(target_coeffs, Y0_coeffs)
      arg_err = net_err / (1. * len(self.cv_coeffs))
      b_errs.append(arg_err)
    self.best_b = self.bandwidths[np.argmin(b_errs)]
  def regress(self, f0, b = None, return_samples = False):
    if (not b): b = self.best_b

    normed_distances = np.array([self.dist_fn(f0, f) for f, _ in self.training_coeffs]) /b
    k_sum = sum([self.kernel(d) for d in normed_distances])
    weights = [self.kernel(normed_distances[i])/k_sum for i in range(len(self.training_coeffs))]
    outputs = [y for _, y in self.training_coeffs]
    a = np.transpose(np.array(outputs))
    b = np.array([[w] for w in weights])
    Y0_coeffs = np.dot(a,b)
    if return_samples:
       density = coeffs_to_approx_density(Y0_coeffs.flatten())
       samples = rejection_sample(0, 1, density, 1000)
       return samples
    return Y0_coeffs.flatten()
class ND_Estimator:
    """
    Nonparametric regression on Fourier coefficient space
    """

    def __init__(
        self,
        training_in_coeffs,
        training_out_coeffs,
        cv_in_coeffs,
        cv_out_coeffs,
        degree=20,
        dim=6,
        dist_fn=L2_distance,
        kernel=RBF_kernel,
        bandwidths=(0.1, 0.2, 0.5, 1, 2),
    ):

        self.degree = degree
        self.dim = dim
        self.dist_fn = dist_fn      # chỉ dùng để đo error
        self.kernel = kernel
        self.bandwidths = list(bandwidths)
        self.best_b = None

        # ÉP VỀ NUMPY ARRAY (quan trọng cho tốc độ + nhất quán)
        self.X_hats = np.asarray(training_in_coeffs)   # (N, D)
        self.Y_hats = np.asarray(training_out_coeffs)  # (N, Dout)
        self.cv_X_hats = np.asarray(cv_in_coeffs)
        self.cv_Y_hats = np.asarray(cv_out_coeffs)

        self.ball_tree = None
    def full_regress(self, f0, b=None):
        """
        Kernel regression using vectorized L2 distance
        """

        if b is None:
            b = self.best_b

        # ===== vectorized L2 distance =====
        # distances: (N,)
        distances = np.linalg.norm(self.X_hats - f0[None, :], axis=1)
        normed_distances = distances / b

        # ===== kernel weights =====
        k_vals = self.kernel(normed_distances)
        k_sum = k_vals.sum()

        if k_sum < 1e-12:
            print(" >>> >>> [debug] WARNING: k_sum underflow detected")
            return np.zeros(self.Y_hats.shape[1])

        weights = k_vals / k_sum

        # ===== weighted average =====
        # returns (Dout,)
        return weights @ self.Y_hats
    def train(self):
        """
        Cross-validate bandwidth
        """

        print(" >>> [debug] cross-validating bandwidths...")
        b_errs = []

        for b in self.bandwidths:
            net_err = 0.0

            for i in range(len(self.cv_X_hats)):
                input_coeffs = self.cv_X_hats[i]
                target_coeffs = self.cv_Y_hats[i]

                Y0_coeffs = self.full_regress(input_coeffs, b=b)
                net_err += self.dist_fn(target_coeffs, Y0_coeffs)

            avg_err = net_err / float(len(self.cv_X_hats))
            print(
                f" >>> >>> [debug] Average L2 error for bandwidth {b} - {avg_err}"
            )
            b_errs.append(avg_err)

        best_idx = int(np.argmin(b_errs))
        self.best_b = self.bandwidths[best_idx]

        print(" >>> >>> [debug] Bandwidth selected:", self.best_b)

        if self.best_b == self.bandwidths[0]:
            print(
                " >>> >>> [debug] WARNING: minimum bandwidth selected. "
                "Consider trying smaller bandwidths."
            )

        if self.best_b == self.bandwidths[-1]:
            print(
                " >>> >>> [debug] WARNING: maximum bandwidth selected. "
                "Consider trying larger bandwidths."
            )
    def build_ball_tree(self):
        self.ball_tree = neighbors.BallTree(self.X_hats)

    def KNN_regress(self, f0, b=None, k=1):

        if b is None:
            b = self.best_b

        distances, indices = self.ball_tree.query(
            np.asarray(f0).reshape(1, -1), k=k
        )
        distances = distances[0]
        indices = indices[0]

        normed_distances = distances / b
        k_vals = self.kernel(normed_distances)
        k_sum = k_vals.sum()

        if k_sum < 1e-12:
            return np.zeros(self.Y_hats.shape[1])

        weights = k_vals / k_sum
        selected_Ys = self.Y_hats[indices]

        return weights @ selected_Ys
