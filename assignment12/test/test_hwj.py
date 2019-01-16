import random
import numpy as np

# a = random.randint(0, 2)
# print(a)

A = np.random.rand(5, 4)
print(A)

print('-' * 100)
A[:, 0:3] = A[:, 1:4]
print(A)
print(A.shape)

B = np.random.rand(5,1)
print(B)
print('-' * 100)
A[:,3] = B[:,0]
print(A)
