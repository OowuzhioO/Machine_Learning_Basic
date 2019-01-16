# import numpy as np
# from sklearn import metrics

# from model.sklearn_multiclass import sklearn_multiclass_prediction
# from model.self_multiclass import MulticlassSVM

# mnist = np.loadtxt('data/mnist_test.csv', delimiter=',')

# X_train = mnist[:len(mnist)//2, 1:]
# y_train = mnist[:len(mnist)//2, 0].astype(np.int)
# X_test = mnist[len(mnist)//2:, 1:]
# y_test = mnist[len(mnist)//2:, 0].astype(np.int)
# y_pred_train, y_pred_test = sklearn_multiclass_prediction('ovr', X_train, y_train, X_test)


# a = 'ovr'
# b = 'ovo'
# c = 'crammer'

# d = 'ovr'
# if d == a:
# 	print(a)
# elif d == b:

# 	print(b)
# elif d == 'crammer':
# 	print(c)
# else:
# 	print("sdfas")

from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
import numpy as np


# a = [0, 1, 2, 3, 4, 5]
# b = np.array(list(a))
# print(b.shape)
# b = b[np.newaxis]
# print(b.shape)
# b = list(a)
# for index in range(len(b)):
# 	if b[index] != 3:
# 		b[index] = 0

# print(b)


# t = np.arange(12).reshape((3,4))
# print(t)
# l = np.empty((1,4))
# print(l.shape)
# print(l)

# for i in range(3):
# 	a = t[i][np.newaxis]
# 	# print(a.shape)
# 	l = np.append(l,a)
# 	print(l.shape)
# print(l.shape)

# a = t[1][np.newaxis]
# l = np.append(a,axis=0)
# print(l.shape)

# a = np.array([[1, 0, 0], [1, 0, 0], [2, 3, 4], [7, 8, 8]])
# b, c = np.unique(a[3], return_counts=True)
# print(b)
# print(c)

# l = [0] * 5
# print(l)
# l[0] = 1
# print(l)
# i =3 
# j =2 

# d = [i, j]
# a = sorted(d)
# print(a)

a = np.array([1,2,3,4,5])
b = np.argmax(a)
print(b)
# for i in a:
# 	print(i)




