"""Implements support vector machine."""

from __future__ import print_function
from __future__ import absolute_import

import numpy as np
from models.linear_model import LinearModel


class SupportVectorMachine(LinearModel):
    """Implements a linear regression mode model"""

    def backward(self, f, y):
        """Performs the backward operation based on the loss in total_loss.

        By backward operation, it means to compute the gradient of the loss
        w.r.t w.

        Hint: You may need to use self.x, and you made need to change the
        forward operation.

        Args:
            f(numpy.ndarray): Output of forward operation, dimension (N,1).
            y(numpy.ndarray): Ground truth label, dimension (N,1).
        Returns:
            total_grad(numpy.ndarray): Gradient of L w.r.t to self.w,
              dimension (ndims+1,).
        """
        reg_grad = None
        loss_grad = None
        # Implementation here.
        reg_grad = self.w_decay_factor * self.w
        indicator = np.dot(y.T, f)
        if indicator < 1:
            loss_grad = np.dot(X.T, y)
        else:
            loss_grad = 0
        total_grad = reg_grad + loss_grad
        return total_loss

    def total_loss(self, f, y):
        """The sum of the loss across batch examples + L2 regularization.
        Total loss is hinge_loss + w_decay_factor/2*||w||^2

        Args:
            f(numpy.ndarray): Output of forward operation, dimension (N,1).
            y(numpy.ndarray): Ground truth label, dimension (N,1).
        Returns:
            total_loss (float): sum hinge loss + reguarlization.
        """

        hinge_loss = None
        l2_loss = None
        # Implementation here.
        N = y.shape[0]
        hinge_loss = 0
        for count in range(N):
            y_i = y[count]
            y_i = float(y_i[0])
            wx_i = f[count]
            wx_i = float(wx_i[0])
            z_i = y_i * wx_i
            hinge_z = max(0, 1 - z_i)
            hinge_loss = hinge_loss + hinge_z

        l2_loss = 0.5 * self.w_decay_factor*np.linalg.norm(self.w, ord=2)** 2
        total_loss = hinge_loss + l2_loss
        return total_loss

    def predict(self, f):
        """Converts score to prediction.

        Args:
            f(numpy.ndarray): Output of forward operation, dimension (N,1).
        Returns:
            (numpy.ndarray): Hard predictions from the score, f,
              dimension (N,1). Tie break 0 to 1.0.
        """
        y_predict = None
        # Implementation here.
        y_predict_helper = []
        for score in f:
            if score < 0:
                label = -1
            else:
                label = 1
            y_predict_helper.append(label)
        y_predict = np.array(y_predict_helper)[np.newaxis]
        y_predict = y_predict.T
        return y_predict