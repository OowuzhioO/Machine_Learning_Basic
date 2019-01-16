from PIL import Image
from skimage import io
# from tensorflow import *
import numpy as np


img_path = '/Users/haowenjiang/Doc/cs/uiuc/Machine_Learning/assignment/assignment4/test/data/image_data/0000.jpg'


img_file1 = Image.open(img_path)
img_file2 = io.imread(img_path)



# print(type(img_file1))
# print(type(img_file2))
# print(img_file2)

# print ("the picture's size: ", img_file1.size)
# print ("the picture's shape: ", img_file2.shape)


# a = 9 
# b = 3
# c = 2
# r = max(a, b, c)
# print(r)

# P_Iden = np.eye(5, dtype=int)
# p = -1 * P_Iden
# p_helper = np.ones((2, 3), dtype=int)
# p[:2, -3:] = p_helper
# print(p)

# from cvxopt  import solvers, matrix 
# P = matrix([[1.0,0.0],[0.0,0.0]])   # matrix里区分int和double，所以数字后面都需要加小数点
# q = matrix([3.0,4.0])
# G = matrix([[-1.0,0.0,-1.0,2.0,3.0],[0.0,-1.0,-3.0,5.0,4.0]])
# h = matrix([0.0,0.0,-15.0,100.0,80.0])

# sol = solvers.qp(P,q,G,h)   # 调用优化函数solvers.qp求解
# print(sol['x'])



a = np.ones((3,4), dtype=np.float32)
print(type(a[0][0]))
