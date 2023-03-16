# [과제]

# 3가지 원핫인코딩 방식을 비교할것

#1. PANDAS의 GET_DUMMIES
#2. KERAS의 TO_CATEGORICAL
#3. SKLEARN의 ONEHOTENCODER

# 미세한 차이를 정리하시오.

# Pandas의 get_dummies:
# Pandas의 get_dummies 함수는 범주형 변수를 더미 변수(dummy variable)로 변환해줍니다. 예를 들어, 'color'라는 열이 있고 그 값으로 'red', 'green', 'blue'가 있는 경우 get_dummies 함수를 사용하면 'color_red', 'color_green', 'color_blue'라는 열을 생성하고 해당 값이 있으면 1, 없으면 0의 값을 가집니다. 이렇게 생성된 더미 변수는 기존 데이터프레임에 새로운 열로 추가됩니다.

# Keras의 to_categorical:
# Keras의 to_categorical 함수는 정수형 레이블(label)을 one-hot 벡터로 변환해줍니다. 예를 들어, 0부터 9까지의 숫자 중 하나를 클래스로 가지는 데이터셋에서 레이블을 정수형으로 표현하고 있다면, to_categorical 함수를 사용하여 이를 one-hot 벡터로 변환할 수 있습니다. 이때, 클래스 개수만큼의 차원을 가진 벡터를 생성하고 해당 클래스에 해당하는 인덱스에 1의 값을 부여하고 나머지는 0의 값을 부여합니다.

# Scikit-learn의 OneHotEncoder:
# Scikit-learn의 OneHotEncoder 클래스는 범주형 변수를 더미 변수로 변환해줍니다. get_dummies 함수와 비슷하게 동작하지만, OneHotEncoder는 fit과 transform 메서드를 사용하여 범주형 변수의 범위를 학습하고 새로운 데이터에 대해 변환합니다. 이는 모델을 학습할 때 훈련 데이터와 테스트 데이터에서 동일한 더미 변수를 생성할 수 있도록 보장해줍니다. 또한, OneHotEncoder는 희소 행렬(sparse matrix)을 사용하여 메모리를 효율적으로 사용할 수 있습니다.