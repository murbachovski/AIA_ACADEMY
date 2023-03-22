import numpy as np

dataset = np.array(range(1, 11))
timesteps = 5
print(range(6))

def split_x(dataset, timesteps):
    aaa = []
    for i in range(len(dataset) - timesteps + 1):
        subset = dataset[i : (i + timesteps)]
        aaa.append(subset)
    return np.array(aaa)
    
bbb = split_x(dataset, timesteps)
print(bbb)
print(bbb.shape) #(6, 5)




