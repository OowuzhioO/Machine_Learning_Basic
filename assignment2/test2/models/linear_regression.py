"""Implements linear regression."""

from __future__ import print_function
from __future__ import absolute_import

import numpy as np
from models.linear_model import LinearModel


class LinearRegression(LinearModel):
    """Implements a linear regression mode model."""

    def backward(self, f, y):
        """Performs the backward operation.

        By backward operation, it means to compute the gradient of the loss
        with respect to w.

        Hint: You may need to use self.x, and you made need to change the
        forward operation.

        Args:
            f(numpy.ndarray): Output of forward operation, dimension (N,1).
            y(numpy.ndarray): Ground truth label, dimension (N,1).

        Returns:
            total_grad(numpy.ndarray): Gradient of L w.r.t to self.w,
              dimension (ndims+1,1).
        """
        total_grad = np.dot(self.x.T, f-y) + self.w_decay_factor * self.w

        return total_grad

    def total_loss(self, f, y):
        """Computes the total loss, square loss + L2 regularization.

        Overall loss is sum of squared_loss + w_decay_factor*l2_loss
        Note: Don't forget the 0.5 in the squared_loss!

        Args:
            f(numpy.ndarray): Output of forward operation, dimension (N,1).
            y(numpy.ndarray): Ground truth label, dimension (N,1).
        Returns:
            total_loss (float): sum square loss + reguarlization.
        """
        loss_matrix = f - y
        l2_loss = np.linalg.norm(self.w, ord=2)** 2
        square_loss_helper = np.linalg.norm(loss_matrix, ord=2)
        square_loss = square_loss_helper** 2
        
        total_loss = 0.5 * (square_loss + self.w_decay_factor*l2_loss)

        return total_loss

    def predict(self, f):
        """Nothing to do here.
        """
        return f