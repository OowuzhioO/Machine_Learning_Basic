"""Semi-supervised learning for EM for GMM."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from utils import io_tools
from models.gaussian_mixture_model import GaussianMixtureModel

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('max_iter', 100, 'Number of EM steps to run.')
flags.DEFINE_integer('n_components', 10, 'Number of components')


def main(_):
    """High level pipeline.

    This scripts performs the training and evaling and testing stages for
    semi-supervised learning using kMeans algorithm.
    """
    # Load dataset.
    _, unlabeled_data = io_tools.read_dataset('data/simple_test.csv')
    n_dims = unlabeled_data.shape[1]

    # Initialize model.
    model = GaussianMixtureModel(n_dims, n_components=FLAGS.n_components,
                                 max_iter=FLAGS.max_iter)

    # Unsupervised training.
    model.fit(unlabeled_data)

    # Supervised training.
    train_label, train_data = io_tools.read_dataset('data/simple_test.csv')
    model.supervised_fit(train_data, train_label)

    # Eval model.
    eval_label, eval_data = io_tools.read_dataset('data/simple_test.csv')
    y_hat_eval = model.supervised_predict(eval_data)

    acc = np.sum(y_hat_eval == eval_label) / (1. * eval_data.shape[0])
    print("Accuracy: %s" % acc)

if __name__ == '__main__':
    tf.app.run()
