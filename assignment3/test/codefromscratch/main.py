"""Main function for binary classifier"""

import numpy as np

from io_tools import *
from logistic_model import *

""" Hyperparameter for Training """
learn_rate = 0.001
max_iters = 1000

if __name__ == '__main__':
    ###############################################################
    # Fill your code in this function to learn the general flow
    # (..., although this funciton will not be graded)
    ###############################################################

    # Load dataset.
    # Hint: A, T = read_dataset('../data/trainset','indexing.txt')
    A, T = read_dataset('../data/trainset','indexing.txt')

    # Initialize model.
    lm = LogisticModel(16)

    # Train model via gradient descent.
    lm.fit(T, A, learn_rate, max_iters)

    # Save trained model to 'trained_weights.np'
    lm.save_model('trained_weights.np')

    # Load trained model from 'trained_weights.np'
    lm.load_model('trained_weights.np')

    # Try all other methods: forward, backward, classify, compute accuracy
    result = lm.classify(A)
    N = T.shape[0]
    count = 0
    for n in range(N):
        if result[n][0] == T[n][0]:
            count = count + 1
    print("The accuracy of scrach is: ", count/N*100, "%")

    
