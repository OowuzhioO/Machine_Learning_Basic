"""Implements feature extraction and other data processing helpers.
(This file will not be graded).
"""

import numpy as np
import skimage
from skimage import color


from io_tools import*


def preprocess_data(data, process_method='default'):
    """Preprocesses dataset.

    Args:
        data(dict): Python dict loaded using io_tools.
        process_method(str): processing methods needs to support
          ['raw', 'default'].
        if process_method is 'raw'
          1. Convert the images to range of [0, 1] by dividing by 255.
          2. Remove dataset mean. Average the images across the batch dimension.
             This will result in a mean image of dimension (8,8,3).
          3. Flatten images, data['image'] is converted to dimension (N, 8*8*3)
        if process_method is 'default':
          1. Convert images to range [0,1]
          2. Convert from rgb to gray then back to rgb. Use skimage
          3. Take the absolute value of the difference with the original image.
          4. Remove dataset mean. Average the absolute value differences across
             the batch dimension. This will result in a mean of dimension (8,8,3).
          5. Flatten images, data['image'] is converted to dimension (N, 8*8*3)

    Returns:
        data(dict): Apply the described processing based on the process_method
        str to data['image'], then return data.
    """
    if process_method == 'raw':
        convert_data = convert_image(data)
        data_mean = dataset_mean(convert_data)
        remove_mean_data = remove_mean(convert_data, data_mean)
        flat_data = flatten_data(remove_mean_data)

        




    elif process_method == 'default':
        convert_data = convert_image(data)
        rgcir_data = rgb_gray_circle(convert_data)
        diff_data = abs_diff(convert_data, rgcir_data)
        data_mean = dataset_mean(diff_data)
        remove_mean_data = remove_mean(convert_data, data_mean)
        flat_data = flatten_data(remove_mean_data)

    elif process_method == 'custom':
        # Design your own feature!
        pass

    data['image'] = flat_data
    return data


def compute_image_mean(data):
    """ Computes mean image.

    Args:
        data(dict): Python dict loaded using io_tools.

    Returns:
        image_mean(numpy.ndarray): Avaerage across the example dimension.
    """
    N = data['image'].shape[0]
    image_mean = None
    image= data['image']
    data_sum = np.sum(image, axis =0)
    image_mean = data_sum / N
    return image_mean


def remove_data_mean(data):
    """Removes data mean.

    Args:
        data(dict): Python dict loaded using io_tools.

    Returns:
        data(dict): Remove mean from data['image'] and return data.
    """
    N = data['image'].shape[0]
    image_mean = compute_image_mean(data)
    for i in range(N):
      data['image'][i] = data['image'][i] - image_mean
    return data


def convert_image(data):
  images = data['image']
  # print(type(data['image']))
  convert_data = images / 255
  return convert_data

def rgb_gray_circle(convert_data):
  gray_image = skimage.color.rgb2gray(convert_data)
  rgb_back_image = skimage.color.gray2rgb(gray_image)
  # print(gray_image)
  return rgb_back_image

def abs_diff(convert_data, rgcir_data):
  diff_data = abs(convert_data - rgcir_data)
  return diff_data

def dataset_mean(diff_data):
  data_mean = np.mean(diff_data, axis=0)
  return data_mean 

def remove_mean(convert_data, data_mean):
  remove_mean_data = convert_data - data_mean
  N = data['image'].shape[0]
  for i in range(N):
        data['image'][i] -= data_mean

  return remove_mean_data

def flatten_data(remove_mean_data):
  flat_data = remove_mean_data.reshape(-1, 8*8*3)
  return flat_data





image_data_path =  '/Users/haowenjiang/Doc/cs/uiuc/Machine_Learning/assignment/assignment4/test/data/image_data'
data_txt_file = '/Users/haowenjiang/Doc/cs/uiuc/Machine_Learning/assignment/assignment4/test/data/test.txt'

data = read_dataset(data_txt_file, image_data_path)
# print(type(data['image']))

# convert_data = convert_image(data)
# rgcir_data = rgb_gray_circle(convert_data)
# diff_data = abs_diff(convert_data, rgcir_data)
# data_mean = dataset_mean(diff_data)
# # remove_mean_data = remove_mean(convert_data, data_mean)
# # flat_data = flatten_data(remove_mean_data)
# print(data_mean.shape)

r = preprocess_data(data)
# print(r['image'])
# print(r['image'][0].shape[0])
# print(r['image'][1].shape)
N = data['label'].shape[0]
ndims = 191
G_yx = np.zeros((N, ndims + 1))
for i in range(N):
  x_i = data['image'][i]
  y_i = data['label'][i]
  y_i = float(y_i[0])
  G_yx[i,:] = -1 * y_i * x_i
print(G_yx)


img_sum = np.sum(img_data, axis =0)

