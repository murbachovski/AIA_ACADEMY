import numpy as np
import pandas as pd
a = [[1, 2, 3,], [4, 5, 6]]
b = np.array(a)
# print(b)
# [[1 2 3]
#  [4 5 6]]

c = [[1,2,3], [4,5]]
# print(c)
# [[1, 2, 3], [4, 5]]

d = np.array(c)
# print(d)
# [list([1, 2, 3]) list([4, 5])]

# 리스트는 크기가 달라도 상관이 없다.
##############################################

e = [[1,2,3], ['바보', '맹구', 5, 6]]
# print(e)
# [[1, 2, 3], ['바보', '맹구', 5, 6]]

# 리스트에는 다른 자료형을 넣어도 상관없다.

f = np.array(e)
# print(f)
# [list([1, 2, 3]) list(['바보', '맹구', 5, 6])]

# print(e.shape)
# AttributeError: 'list' object has no attribute 'shape'
