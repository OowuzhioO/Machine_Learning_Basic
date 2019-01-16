"""Implements the Gaussian Mixture model, and trains using EM algorithm."""
import numpy as np
import scipy
from scipy.stats import multivariate_normal


class GaussianMixtureModel(object):
    """Gaussian Mixture Model"""

    def __init__(self, n_dims, n_components=1,
                 max_iter=10,
                 reg_covar=1e-6):
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
        # np.array of size (n_components, 1)
        self._pi = np.ones((self._n_components, 1)) / self._n_components

        # Initialized with identity.
        sigma_helper = []
        for i in range(self._n_components):
            sigma_helper.append(1000 * np.eye(self._n_dims))
        # np.array of size (n_components, n_dims, n_dims)
        self._sigma = np.array(sigma_helper)

    def fit(self, x):
        """Runs EM steps.

        Runs EM steps for max_iter number of steps.

        Args:
            x(numpy.ndarray): Feature array of dimension (N, ndims).
        """
        N = x.shape[0]
        u_k_collection = []
        for i in range(self._n_components):
            curr = np.random.randint(N)
            u_k_collection.append(x[curr])
        self._mu = np.array(u_k_collection)
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
        n_components = z_ik.shape[1]
        Nk = np.sum(z_ik, axis=0)
        self._pi = Nk / N
        mu_k_collection = []
        for k in range(n_components):
            mu_k = []
            # mu_k_temp = 0
            for i in range(N):
                mu_k.append(z_ik[i, k] * x[i])
                # mu_k_temp = mu_k_temp + z_ik[i, k] * x[i]
            mu_k = np.array(mu_k)
            mu_k_sum = np.sum(mu_k, axis=0) / Nk[k]
            mu_k_collection.append(mu_k_sum)
        self._mu = np.array(mu_k_collection)

        sigma_k_collection = []
        for k in range(n_components):
            sigma_k = []
            for i in range(N):
                x_mu = x[i] - self._mu[k]
                x_mu = x_mu[np.newaxis].T
                sigma_k_helper = z_ik[i,k] * np.dot(x_mu, x_mu.T)
                sigma_k.append(sigma_k_helper)
            sigma_k = np.array(sigma_k)
            sigma_k_sum = np.sum(sigma_k, axis=0) / Nk[k]
            sigma_k_sum = sigma_k_sum + self._reg_covar * np.eye(self._n_dims)
            sigma_k_collection.append(sigma_k_sum)
        self._sigma = np.array(sigma_k_collection)

    def get_conditional(self, x):
        """Computes the conditional probability.

        p(x^(i)|z_ik=1)

        Args:
            x(numpy.ndarray): Feature array of dimension (N, ndims).
        Returns:
            ret(numpy.ndarray): The conditional probability for each example,
                dimension (N,, n_components).
        """
        # ret = None
        ret = []
        for i in range(self._n_components):
            mu_k = self._mu[i]
            sigma_k = self._sigma[i]
            ret.append(self._multivariate_gaussian(x, mu_k, sigma_k))

        return np.array(ret).T

    def get_marginals(self, x):
        """Computes the marginal probability.

        p(x^(i)|pi, mu, sigma)

        Args:
             x(numpy.ndarray): Feature array of dimension (N, ndims).
        Returns:
            (1) The marginal probability for each example, dimension (N,).
        """
        cond = self.get_conditional(x)
        marg = np.dot(cond, self._pi)
        return marg

    def get_posterior(self, x):
        """Computes the posterior probability.

        p(z_{ik}=1|x^(i))

        Args:
            x(numpy.ndarray): Feature array of dimension (N, ndims).
        Returns:
            z_ik(numpy.ndarray): Array containing the posterior probability
                of each example, dimension (N, n_components).
        """
        cond = self.get_conditional(x)
        w_cond = cond * (self._pi.T)
        marg = self.get_marginals(x)[np.newaxis].T
        z_ik = np.squeeze(w_cond / marg)
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

        self.cluster_label_map = []
        N = x.shape[0]
        self.fit(x)
        z_ik = self.get_posterior(x)
        cluster = np.argmax(z_ik, axis=1)

        for i in range(self._n_components):
            ref_index = np.argwhere(cluster == i)
            if ref_index.size != 0:
                y_ref_labels = y[ref_index].flatten()
                y_ref_labels = y_ref_labels.tolist()
                for label in y_ref_labels:
                    num_helper = y_ref_labels.count(label)
                    num = max(y_ref_labels.count(m) for m in y_ref_labels)
                    if num == num_helper:
                        target = label
                self.cluster_label_map.append(target)
            else:
                self.cluster_label_map.append(i)

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
        index = np.argmax(z_ik, axis=1)
        y_hat = []
        for i in range(N):
            y_hat.append(self.cluster_label_map[index[i]])

        return np.array(y_hat)
