"""Main function for binary classifier"""
import tensorflow as tf
import numpy as np
from io_tools import *
from logistic_model import *

""" Hyperparameter for Training """
learn_rate = 0.001
max_iters = 1000

def main(_):
    ###############################################################
    # Fill your code in this function to learn the general flow
    # (..., although this funciton will not be graded)
    ###############################################################

    # Load dataset.
    # Hint: A, T = read_dataset_tf('../data/trainset','indexing.txt')
    A, T = read_dataset_tf('../data/trainset','indexing.txt')

    # Initialize model.
    lm_tf = LogisticModel_TF(16)

    # Build TensorFlow training graph
    lm_tf.build_graph(learn_rate)
    
    # Train model via gradient descent.
    result = lm_tf.fit(T, A, max_iters)

    # Compute classification accuracy based on the return of the "fit" method
    N = T.shape[0]
    count = 0
    for n in range(N):
        if result[n][0] == T[n][0]:
            count = count + 1
    print("The accuracy of tf is: ", count/N*100, '%')
    
if __name__ == '__main__':
    tf.app.run()