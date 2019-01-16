"""Input and output helpers to load in data.
"""
import numpy as np
# import tensorflow as tf


def read_dataset(path_to_dataset_folder, index_filename):
    """ Read dataset into numpy arrays with preprocessing included
    Args:
        path_to_dataset_folder(str): path to the folder containing samples and indexing.txt
        index_filename(str): indexing.txt
    Returns:
        A(numpy.ndarray): sample feature matrix A = [[1, x1], 
                                                     [1, x2], 
                                                     [1, x3],
                                                     .......] 
                                where xi is the 16-dimensional feature of each sample

        T(numpy.ndarray): class label vector T = [y1, y2, y3, ...] 
                             where yi is +1/-1, the label of each sample 
    """
    ###############################################################
    # Fill your code in this function
    ###############################################################
    # Hint: open(path_to_dataset_folder+'/'+index_filename,'r')
    with open(path_to_dataset_folder + '/' + index_filename, 'r') as index_file:
        lines = index_file.readlines()
        txt_paths = {}
        for line in lines:
            txt_path = line.split()
            txt_paths[txt_path[1]] = txt_path[0]

    A = []
    T = []
    for sample_file, label in txt_paths.items():
        A_16dim = []
        with open(path_to_dataset_folder + '/' + sample_file, 'r') as dim_values:
            lines = dim_values.readlines()
            for line in lines:
                dim_value = line.split()
                A_helper = [1]
                for element in dim_value:
                    A_helper.append(float(element))
            A.append(A_helper)
            label = int(label)
            T_helper = [label]
            T.append(T_helper)
    A = np.array(A)
    T = np.array(T)
    return A, T