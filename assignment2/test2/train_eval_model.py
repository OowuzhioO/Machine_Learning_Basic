"""
Train model and eval model helpers.
"""
from __future__ import print_function

import numpy as np
from models.linear_regression import LinearRegression


def train_model(processed_dataset, model, learning_rate=0.001, batch_size=16,
                num_steps=1000, shuffle=True):
    """Implements the training loop of stochastic gradient descent.

    Performs stochastic gradient descent with the indicated batch_size.
    If shuffle is true:
        Shuffle data at every epoch, including the 0th epoch.
    If the number of example is not divisible by batch_size, the last batch
    will simply be the remaining examples.

    Args:
        processed_dataset(list): Data loaded from io_tools
        model(LinearModel): Initialized linear model.
        learning_rate(float): Learning rate of your choice
        batch_size(int): Batch size of your choise.
        num_steps(int): Number of steps to run the updated.
        shuffle(bool): Whether to shuffle data at every epoch.
    Returns:
        model(LinearModel): Returns a trained model.
    """
    # Perform gradient descent.
    if shuffle is True:
        i = 0
        sizeofds = processed_dataset[0].shape[0] 
        shuf = np.arange(sizeofds)
        np.random.shuffle(shuf)
        processed_dataset[0] = processed_dataset[0][shuf]
        processed_dataset[1] = processed_dataset[1][shuf]
        for step in range(num_steps):
            if i + batch_size <= sizeofds:
                x_batch = processed_dataset[0][i:i+batch_size,:]
                y_batch = processed_dataset[1][i:i+batch_size,:]
                update_step(x_batch, y_batch, model, learning_rate)
                i = i + batch_size
            else:
                x_batch = processed_dataset[0][i:,:]
                y_batch = processed_dataset[1][i:,:]
                update_step(x_batch, y_batch, model, learning_rate)
                i = 0
                shuf = np.arange(sizeofds)
                np.random.shuffle(shuf)
                processed_dataset[0] = processed_dataset[0][shuf]
                processed_dataset[1] = processed_dataset[1][shuf]
    return model    
    


def update_step(x_batch, y_batch, model, learning_rate):
    """Performs on single update step, (i.e. forward then backward).

    Args:
        x_batch(numpy.ndarray): input data of dimension (N, ndims).
        y_batch(numpy.ndarray): label data of dimension (N, 1).
        model(LinearModel): Initialized linear model.
    """
    forw = model.forward(x_batch)
    back = model.backward(forw, y_batch)
    model.w = model.w - learning_rate * back


def train_model_analytic(processed_dataset, model):
    """Computes and sets the optimal model weights (model.w).

    Args:
        processed_dataset(list): List of [x,y] processed
            from utils.data_tools.preprocess_data.
        model(LinearRegression): LinearRegression model.
    """
    x = processed_dataset[0]
    a=np.ones(shape=(x.shape[0],1))
    x=np.append(x,a,axis=1)
    y = processed_dataset[1]
    ident = np.eye(x.shape[1])
    help1 = np.linalg.inv(np.dot(x.T, x) + model.w_decay_factor * ident)
    help2 = np.dot(help1, x.T)
    model.w = np.dot(help2, y)


def eval_model(processed_dataset, model):
    """Performs evaluation on a dataset.

    Args:
        processed_dataset(list): Data loaded from io_tools.
        model(LinearModel): Initialized linear model.
    Returns:
        loss(float): model loss on data.
        acc(float): model accuracy on data.
    """
    f = model.forward(processed_dataset[0])
    loss = model.total_loss(f, processed_dataset[1])

    return loss