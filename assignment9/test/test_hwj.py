import numpy as np
A = np.random.randint(5, size=3)
# print(A.shape)


# A = np.random.randint(5, size=(3,4,2))
# print(A)
# B = A[np.newaxis]
# print(B.shape)

# a = [0, 1, 2, 1, 2, 1]
# for n in a:

#     old = a.count(n)

#     num = max(a.count(m) for m in a)
# # print(num)
#     if num == old:
#         target = n
# print(target)


# data/simple_test.csv


A = [0, 1, 2, 1, 2, 1, 3, 5, 7, 7, 7, 8, 9, 4, 7, 1]
A = np.array(A)
# # # print(A.shape)
# unique, counts = np.unique(A, return_counts=True)
# max_ind = np.argmax(counts)
# max_cluster = unique[max_ind]
# print(max_cluster)


y_ref_labels = A
y_ref_labels = y_ref_labels.tolist()
for label in y_ref_labels:
    num_helper = y_ref_labels.count(label)
    num = max(y_ref_labels.count(m) for m in y_ref_labels)
    if num == num_helper:
        target = label
        break
print(target)
