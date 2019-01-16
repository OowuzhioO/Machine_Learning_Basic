"""Input and output helpers to load in data.
(This file will not be graded.)
"""

import numpy as np
import skimage
import os
from skimage import io


def read_dataset(data_txt_file, image_data_path):
    """Read data into a Python dictionary.

    Args:
        data_txt_file(str): path to the data txt file.
        image_data_path(str): path to the image directory.

    Returns:
        data(dict): A Python dictionary with keys 'image' and 'label'.
            The value of dict['image'] is a numpy array of dimension (N,8,8,3)
            containing the loaded images.

            The value of dict['label'] is a numpy array of dimension (N,1)
            containing the loaded label.

            N is the number of examples in the data split, the exampels should
            be stored in the same order as in the txt file.
    """
    data = {}
    data['image'] = None
    data['label'] = None

    image_list = []
    # c1 = 0
    # c2 = 0
    for file in os.listdir(image_data_path):
        if os.path.splitext(file)[1] == '.jpg':
            image_list.append(file)
    image_list.sort()

    image_list_correct = []
    image_label_list = []
    with open(data_txt_file) as data_file:
        lines = data_file.readlines()
        for line in lines:
            line = line.strip()
            image_txt_index = line.split(',')[0]
            for image in image_list:
                if image_txt_index == image.split('.')[0]:
                    image_list_correct.append(image)
                    # print(image)
                    image_label_list.append(float(line.split(',')[1]))
    data_key = []
    data_value = image_label_list

    for ilc in image_list_correct:
        image_path = os.path.join(image_data_path, ilc)
        image_file = io.imread(image_path)
        data_key.append(image_file)
        # print(image_file)
    data_key_np = np.array(data_key)
    # print(data_key_np.shape)
    data_value_np = np.array(data_value)[np.newaxis].T
    # print(data_value_np)
    data['image'] = data_key_np
    data['label'] = data_value_np
    # print(type(data['image']))
    # print(data)
    return data



image_data_path =  '/Users/haowenjiang/Doc/cs/uiuc/Machine_Learning/assignment/assignment4/test/data/image_data'
data_txt_file = '/Users/haowenjiang/Doc/cs/uiuc/Machine_Learning/assignment/assignment4/test/data/test.txt'

data = read_dataset(data_txt_file, image_data_path)
# print(data)
