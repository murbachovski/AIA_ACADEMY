import numpy as np

datasets = np.array(range(1, 41)).reshape(10, 4) # 1 ~ 40 vector형태(1차원)
print(datasets)
print(datasets.shape) #(10, 4)

# x_data = datasets[:, :3]
x_data = datasets[:, :-1]
y_data = datasets[:, -1]
print(x_data)
print(y_data)
print(x_data.shape)         #(10, 3)
print(y_data.shape)         #(10,)

timesteps = 6
print(len(datasets))
###############<MAKE A X>#################
def split_x(dataset, timesteps):
    aaa = []
    for i in range(len(dataset) - timesteps):
        subset = dataset[i : (i + timesteps)]
        aaa.append(subset)
    return np.array(aaa)
    
x_data = split_x(x_data, timesteps)
print(x_data)
# [[[ 1  2  3]
#   [ 5  6  7]
#   [ 9 10 11]
#   [13 14 15]
#   [17 18 19]]

#  [[ 5  6  7]
#   [ 9 10 11]
#   [13 14 15]
#   [17 18 19]
#   [21 22 23]]

#  [[ 9 10 11]
#   [13 14 15]
#   [17 18 19]
#   [21 22 23]
#   [25 26 27]]

#  [[13 14 15]
#   [17 18 19]
#   [21 22 23]
#   [25 26 27]
#   [29 30 31]]

#  [[17 18 19]
#   [21 22 23]
#   [25 26 27]
#   [29 30 31]
#   [33 34 35]]]
print(x_data.shape) #(5, 5, 3)

###############<MAKE A X>#################
y_data = y_data[timesteps:]
print(y_data) #[24 28 32 36 40]