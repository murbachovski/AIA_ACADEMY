import numpy as np

x1 = np.array([[1, 2], [3, 4], [5, 6]])
x2 = np.array([[[1, 2, 3]], [[4, 5, 6]]])
x3 = np.array([[[1, 2]]])
x4 = np.array([[1], [2], [3], [4], [5]])
x5 = np.array([[[1]], [[2]], [[3]], [[4]]])
x6 = np.array([
    [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
    ])
x7 = np.array([
    [[1, 2]], [[3, 4]], [[5, 6]], [[7, 8]], [[9, 10]]
    ]) 
x8 = np.array([[[1, 2]], [[3, 4]], [[5, 6]]]) 
x9 = np.array([[[1, 2]], [[3, 4]]]) 
x10 = np.array([[1, 2], [3, 4]])
x11 = np.array(
    [[[[[1, 2, 3]], [[5, 6, 7]], [[8, 9, 10]], [[11, 12, 13]]]]] #4, 3
)
x12 = np.array([[1, 2, 3],
               [5, 6, 7],
               [8, 9, 10],
               [11, 12, 13]]
               )
print('x1.shape:', x1.shape)
print("@@@@@@@@@@@@@@@@@@@@@@@@@")
print('x2.shape:',x2.shape) 
print('x3.shape:',x3.shape) 
print('x4.shape:',x4.shape) 
print('x5.shape:',x5.shape) 
print('x6.shape:',x6.shape) 
print('x7.shape:',x7.shape) 
print('x8.shape:',x8.shape) 
print('x9.shape:',x9.shape) 
print('x10.shape:',x10.shape) 
print('x11.shape:', x11.shape)
print('x12.shape:', x12.shape)
# git commit test