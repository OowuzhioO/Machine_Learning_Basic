"""
Train model and eval model helpers.
"""
from __future__ import print_function

import numpy as np
import cvxopt
import cvxopt.solvers


def train_model(data, model, learning_rate=0.001, batch_size=16,
                num_steps=1000, shuffle=True):
    """Implements the training loop of stochastic gradient descent.

    Performs stochastic gradient descent with the indicated batch_size.

    If shuffle is true:
        Shuffle data at every epoch, including the 0th epoch.

    If the number of example is not divisible by batch_size, the last batch
    will simply be the remaining examples.

    Args:
        data(dict): Data loaded from io_tools
        model(LinearModel): Initialized linear model.
        learning_rate(float): Learning rate of your choice
        batch_size(int): Batch size of your choise.
        num_steps(int): Number of steps to run the updated.
        shuffle(bool): Whether to shuffle data at every epoch.

    Returns:
        model(LinearModel): Returns a trained model.
    """

    # Performs gradient descent. (This function will not be graded.)
    if shuffle is True:
        i = 0
        sizeofds = data['image'].shape[0]
        shuf = np.arange(sizeofds)
        np.random.shuffle(shuf)
        data['image'] = data['image'][shuf]
        data['label'] = data['label'][shuf]
        for step in num_steps:
            if i + batch_size <=sizeofds:
                x_batch = data['image'][i:i+batch_size,:]
                y_batch = data['label'][i:i+batch_size,:]
                update_step(x_batch, y_batch, model, learning_rate)
                i = i + batch_size
            else:
                x_batch = data['image'][i:,:]
                y_batch = data['label'][i:,:]
                update_step(x_batch, y_batch, model, learning_rate)
                i = 0
                shuf = np.arange(sizeofds)
                np.random.shuffle(shuf)
                data['image'] = data['image'][shuf]
                data['label'] = data['label'][shuf]

    return model


def update_step(x_batch, y_batch, model, learning_rate):
    """Performs on single update step, (i.e. forward then backward).

    Args:
        x_batch(numpy.ndarray): input data of dimension (N, ndims).
        y_batch(numpy.ndarray): label data of dimension (N, 1).
        model(LinearModel): Initialized linear model.
    """
    # Implementation here. (This function will not be graded.)
    forw = model.forward(x_batch)
    back = model.backward(forw, y_batch)
    model.w = model.w - learning_rate * back


def train_model_qp(data, model):
    """Computes and sets the optimal model wegiths (model.w) using a QP solver.

    Args:
        data(dict): Data from utils.data_tools.preprocess_data.
        model(SupportVectorMachine): Support vector machine model.
    """
    P, q, G, h = qp_helper(data, model)
    P = cvxopt.matrix(P, P.shape, 'd')
    q = cvxopt.matrix(q, q.shape, 'd')
    G = cvxopt.matrix(G, G.shape, 'd')
    h = cvxopt.matrix(h, h.shape, 'd')
    sol = cvxopt.solvers.qp(P, q, G, h)
    z = np.array(sol['x'])
    # Implementation here (do not modify the code above)
    # Set model.w
    model.w = z[:model.ndims + 1, :]


def qp_helper(data, model):
    """Prepares arguments for the qpsolver.

    Args:
        data(dict): Data from utils.data_tools.preprocess_data.
        model(SupportVectorMachine): Support vector machine model.

    Returns:
        P(numpy.ndarray): P matrix in the qp program.
        q(numpy.ndarray): q matrix in the qp program.
        G(numpy.ndarray): G matrix in the qp program.
        h(numpy.ndarray): h matrix in the qp program.
    """
    P = None
    q = None
    G = None
    h = None
    # Implementation here.
    N = data['label'].shape[0]
    P = np.zeros((N + model.ndims + 1, N + model.ndims + 1))
    P_Iden = np.eye(model.ndims + 1, dtype=int)
    P[:model.ndims+1, :model.ndims+1] = P_Iden
    q = np.zeros((1, N + model.ndims + 1))
    q_ones = np.ones((1, N))
    q[:,-N:] = q_ones
    G = np.zeros((2 * N, N + model.ndims + 1))
    G_neg_ones = -1 * np.eye(N, dtype=int)
    G_yx = np.zeros((N, model.ndims + 1))
    x_cur = np.ones((N, ndims + 1))
    x_cur[:,:-1] = data['image']
    for i in range(N):
        x_i = data['image'][i]
        y_i = data['label'][i]
        y_i = float(y_i[0])
        G_yx[i,:] = -1 * y_i * x_i
    G[:N, -N:] = G_neg_ones
    G[N:,:model.ndims+1] = G_yx
    G[N:-N:] = G_neg_ones
    h = np.zeros((2*N, 1))
    h_neg_ones = -1 * np.ones((N, 1))
    h[N:,:] = h_neg_ones

    return P, q, G, h


def eval_model(data, model):
    """Performs evaluation on a dataset.

    Args:
        data(dict): Data loaded from io_tools.
        model(LinearModel): Initialized linear model.

    Returns:
        loss(float): model loss on data.
        acc(float): model accuracy on data.
    """
    # Implementation here.
    loss = 0
    acc = 0
    f = model.forward(data['image'])
    loss = model.total_loss(f, data['label'])
    predict_label = model.predict(f)
    N = data['label'].shape[0]
    count = 0
    for n in range(N):
        if predict_label[n][0] == data['label'][n][0]:
            count = count + 1
    acc = count / N

    return loss, acc