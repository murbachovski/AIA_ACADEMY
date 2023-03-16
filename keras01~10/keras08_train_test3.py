import numpy as np  
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense  
# from sklearn.model_selection import train_test_split


x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size= 0.3, shuffle=True, random_state=None)

# print(x_train)
# print(y_train)
# print(x_valid)
# print(y_valid)
#왜 비율이 7:3으로 잡힐까? = Default 값이 7:3인듯 하다 / test_size로 비율을 설정할 수 있다. / random_state는 셔플 자체에 대한 random값이다 None으로 넣어주면 계속 random으로 1을 넣어주면 1로 들어갔던 값으로 들어간다.

# [검색] train and test data shuffle 7:3 how to get
# hint 싸이킷런

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    # train_size=0.7,
    test_size=0.3,
    random_state=1234,
    # shuffle=False,
)
print(x_train)
print(x_test)

#2. MODEL
model = Sequential()
model.add(Dense(1, input_dim = 1))

#3. COMPILE
model.compile(loss = 'mse', optimizer = 'adam') 
model.fit(x_train, y_train, epochs = 100, batch_size = 1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss:', loss)
result = model.predict([11])
print('[11]predict:', result)