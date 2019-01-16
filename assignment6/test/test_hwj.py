import numpy as np

num_itr = 1000

# define batch size.
batchSize = 3.

# define the input data dimension.
inputSize = 2

# define the output dimension.
outputSize = 1

# define the dimension of the hidden layer.
hiddenSize = 3



# n = np.arange(6).reshape(2,3)
# sig = 1/(1+np.exp(-n))
# print(sig.shape)
# sig = np.sum(sig, axis=0)


# print(sig.shape)

b = 0.2
r = b - b**2
# r = b * (1-b)
print(r)

c = 0.2**2
print(c)