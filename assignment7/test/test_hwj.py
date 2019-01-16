import numpy as np

# A = np.zeros((5,6))
# A[2][3] = 1

# print(A)

# for i in range(5):
# 	for j in range(6):
# 		if A[i][j] == 1:
# 			# print



# B = [(0,0), (1, 1), (2,3), (5,6), (1,9)]

# for i in B:
# 	if i == (2,3):
# 		index = B.index((2,3))
# print(index)

# a = 6
# b = 5
# c = (a, b)
# c = sorted(c)
# print(c)

# a = np.zeros((5,2), dtype=int)
# a[:,1] = 1
# # print(a)
# b = [[1,0], [0,1], [0,1], [1,0], [0,1]]
# b = np.array(b)
# # print(a)
# # print(b)
# # print(a*b)
# d = a*b
# c = np.sum(d)
# print(c)

a = np.arange(6)
np.random.shuffle(a)
print(a)