import numpy as np

a = np.array([[1, 2, 3], [6, 4, 5], [7, 9, 2], [3, 2, 1], [2, 3, 1], [1], [4]]) #2차원
print(a)
print(a.shape)
print(np.argmax(a)) # 전체 데이터 중에서 가장 큰 자리의 index가 나온다. = 7 = 9
print(np.argmax(a, axis=0))  # axis=0 = 행끼리 비교합니다. [2 2 1]
print(np.argmax(a, axis=1))  # axis=1 = 열끼리 비교합니다. [2 0 1 0 1]
print(np.argmax(a, axis=-1)) # axis=-1 = 가장 마지막 축은 1차원 #그래서 -1을 쓰면 이 데이터는 1과 동일
