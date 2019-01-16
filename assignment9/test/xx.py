"""Implements the Gaussian Mixture model, and trains using EM algorithm."""
import numpy as np
import scipy
from scipy.stats import multivariate_normal


class GaussianMixtureModel(object):
    """Gaussian Mixture Model"""

    def __init__(self, n_dims, n_components=1,
                 max_iter=10,
                 reg_covar=1000):
        """
        Args:
            n_dims: The dimension of the feature.
            n_components: Number of Gaussians in the GMM.
            max_iter: Number of steps to run EM.
            reg_covar: Amount to regularize the covariance matrix, (i.e. add
                to the diagonal of covariance matrices).
        """
        self._n_dims = n_dims
        self._n_components = n_components
        self._max_iter = max_iter
        self._reg_covar = reg_covar

        # Randomly Initialize model parameters
        # np.array of size (n_components, n_dims)
        self._mu = np.zeros((self._n_components, self._n_dims))

        # Initialized with uniform distribution.
        one_temp = np.ones((self._n_components, 1))
        # np.array of size (n_components, 1)
        self._pi = one_temp / self._n_components

        # Initialized with identity.
        sig_temp = []
        eye_temp = 1000 * np.eye(self._n_dims)
        for i in range(self._n_components):
            sig_temp.append(eye_temp)

        # np.array of size (n_components, n_dims, n_dims)
        self._sigma = np.array(sig_temp)

    def fit(self, x):
        """Runs EM steps.

        Runs EM steps for max_iter number of steps.

        Args:
            x(numpy.ndarray): Feature array of dimension (N, ndims).
        """
        N = x.shape[0]
        init_temp = np.random.randint(N, size=self._n_components)
        mu_init_temp = []
        for i in range(self._n_components):
            mu_init_temp.append(x[init_temp[i]])
        self._mu = np.array(mu_init_temp)

        for i in range(self._max_iter):
            z_ik = self._e_step(x)
            self._m_step(x, z_ik)

    def _e_step(self, x):
        """E step.

        Wraps around get_posterior.

        Args:
            x(numpy.ndarray): Feature array of dimension (N, ndims).
        Returns:
            z_ik(numpy.ndarray): Array containing the posterior probability
                of each example, dimension (N, n_components).
        """
        z_ik = self.get_posterior(x)
        return z_ik

    def _m_step(self, x, z_ik):
        """M step, update the parameters.

        Args:
            x(numpy.ndarray): Feature array of dimension (N, ndims).
            z_ik(numpy.ndarray): Array containing the posterior probability
                of each example, dimension (N, n_components).
                (Alternate way of representing categorical distribution of z_i)
        """
        # Update the parameters.
        N = x.shape[0]
        #print("N is:", N)
        K = z_ik.shape[1]
        nk = np.sum(z_ik, axis=0)
        self._pi = nk / N
        #print("z_ik is: ",z_ik.shape)

        mu_temp = []
        for k in range(K):
            zx_temp = []
            for i in range(N):
                zx_temp.append(z_ik[i, k] * x[i])
            zx_temp = np.array(zx_temp)
            zx_resu = np.sum(zx_temp, axis=0)
            zx_resu = zx_resu / nk[k]
            mu_temp.append(zx_resu)
        self._mu = np.array(mu_temp)

        sigma_temp = []
        for k in range(K):
            sig_temp = []
            for i in range(N):
                xmu_temp = x[i] - self._mu[k]
                xmu_temp = xmu_temp[np.newaxis].T
                x_temp = z_ik[i, k] * np.dot(xmu_temp, xmu_temp.T)
                sig_temp.append(x_temp)
            sig_temp = np.array(sig_temp)
            sig_resu = np.sum(sig_temp, axis=0)
            sig_resu = sig_resu / nk[k]
            sig_resu += self._reg_covar * np.eye(self._n_dims)
            sigma_temp.append(sig_resu)
        self._sigma = np.array(sigma_temp)

    def get_conditional(self, x):
        """Computes the conditional probability.

        p(x^(i)|z_ik=1)

        Args:
            x(numpy.ndarray): Feature array of dimension (N, ndims).
        Returns:
            ret(numpy.ndarray): The conditional probability for each example,
                dimension (N, n_components).
        """
        #ret = None
        ret = []
        for i in range(self._n_components):
            mu_temp = self._mu[i]  # [np.newaxis]
            #mu_temp = mu_temp.T
            sigma_temp = self._sigma[i]
            ret.append(self._multivariate_gaussian(x, mu_temp, sigma_temp))
        return np.array(ret).T

    def get_marginals(self, x):
        """Computes the marginal probability.

        p(x^(i)|pi, mu, sigma)

        Args:
             x(numpy.ndarray): Feature array of dimension (N, ndims).
        Returns:
            (1) The marginal probability for each example, dimension (N,).
        """
        condi_prob = self.get_conditional(x)
        margin_prob = np.dot(condi_prob, self._pi)
        return margin_prob

    def get_posterior(self, x):
        """Computes the posterior probability.

        p(z_{ik}=1|x^(i))

        Args:
            x(numpy.ndarray): Feature array of dimension (N, ndims).
        Returns:
            z_ik(numpy.ndarray): Array containing the posterior probability
                of each example, dimension (N, n_components).
        """
        condi_prob = self.get_conditional(x)
        weighted_condi_prob = condi_prob * (self._pi.T)
        margin_prob = self.get_marginals(x)
        margin_prob = margin_prob[np.newaxis].T

        z_ik = weighted_condi_prob / margin_prob
        z_ik = np.squeeze(z_ik)
        return z_ik

    def _multivariate_gaussian(self, x, mu_k, sigma_k):
        """Multivariate Gaussian, implemented for you.
        Args:
            x(numpy.ndarray): Array containing the features of dimension (N,
                ndims)
            mu_k(numpy.ndarray): Array containing one single mean (ndims,1)
            sigma_k(numpy.ndarray): Array containing one signle covariance matrix
                (ndims, ndims)
        """
        return multivariate_normal.pdf(x, mu_k, sigma_k)

    def supervised_fit(self, x, y):
        """Assign each cluster with a label through counting.
        For each cluster, find the most common digit using the provided (x,y)
        and store it in self.cluster_label_map.
        self.cluster_label_map should be a list of length n_components,
        where each element maps to the most common digit in that cluster.
        (e.g. If self.cluster_label_map[0] = 9. Then the most common digit
        in cluster 0 is 9.
        Args:
            x(numpy.ndarray): Array containing the feature of dimension (N,
                ndims).
            y(numpy.ndarray): Array containing the label of dimension (N,)
        """
        N = x.shape[0]
        self.fit(x)
        z_ik = self.get_posterior(x)
        cluster_temp = np.argmax(z_ik, axis=1)

        self.cluster_label_map = []
        for i in range(self._n_components):
            arg_temp = np.argwhere(cluster_temp == i)
            if arg_temp.size == 0:
                self.cluster_label_map.append(i)
            else:
                unique, counts = np.unique(y[arg_temp], return_counts=True)
                max_ind = np.argmax(counts)
                max_cluster = unique[max_ind]
                self.cluster_label_map.append(max_cluster)

    def supervised_predict(self, x):
        """Predict a label for each example in x.
        Find the get the cluster assignment for each x, then use
        self.cluster_label_map to map to the corresponding digit.
        Args:
            x(numpy.ndarray): Array containing the feature of dimension (N,
                ndims).
        Returns:
            y_hat(numpy.ndarray): Array containing the predicted label for each
            x, dimension (N,)
        """
        N = x.shape[0]
        z_ik = self.get_posterior(x)
        cluster_ind = np.argmax(z_ik, axis=1)
        y_hat = []
        for i in range(N):
            y_hat.append(self.cluster_label_map[cluster_ind[i]])

        return np.array(y_hat)
