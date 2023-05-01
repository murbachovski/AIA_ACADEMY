import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import time
import datetime

# PATH
path='./_data/dacon_kcal/'
save_path= './_save/dacon_kcal/'

# CALL DATA
train_df = pd.read_csv(path + 'train.csv')
test_df = pd.read_csv(path + 'test.csv')
sample_submission_df = pd.read_csv(path + 'sample_submission.csv')

# ID COLUMN REMOVE
train_df = train_df.drop('ID', axis=1)
test_df = test_df.drop('ID', axis=1)

# Weight_Status, Gender => NUMBER
train_df['Weight_Status'] = train_df['Weight_Status'].map({'Normal Weight': 0, 'Overweight': 1, 'Obese': 2})
train_df['Gender'] = train_df['Gender'].map({'M': 0, 'F': 1})
test_df['Weight_Status'] = test_df['Weight_Status'].map({'Normal Weight': 0, 'Overweight': 1, 'Obese': 2})
test_df['Gender'] = test_df['Gender'].map({'M': 0, 'F': 1})

# PolynomialFeatures DATA 
poly = PolynomialFeatures(degree=2, include_bias=False)
X = poly.fit_transform(train_df.drop('Calories_Burned', axis=1))
y = train_df['Calories_Burned']

# SCALER
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# train, valid SPLIT
X_train, X_valid, y_train, y_valid = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# MLPRegressor는 입력층, 은닉층, 출력층으로 구성되어 있으며, 각 층은 여러 개의 뉴런으로 구성됩니다. 입력층은 입력 데이터를 받아들이고, 은닉층은 입력 데이터를 처리하여 출력값을 계산하며, 출력층은 최종적으로 모델의 출력값을 생성합니다.

# MLPRegressor는 각 뉴런의 가중치와 편향을 조정하여 학습을 진행합니다. 학습 과정에서는 입력 데이터와 출력 데이터를 사용하여 오차를 계산하고, 이 오차를 최소화하기 위해 가중치와 편향을 업데이트합니다.

# MLPRegressor 모델의 성능을 높이기 위해서는 여러 가지 하이퍼파라미터를 조정할 수 있습니다. 예를 들어, 은닉층의 수와 뉴런의 수, 활성화 함수, 최적화 알고리즘, 학습률 등을 조정하여 최적의 성능을 얻을 수 있습니다. 또한, 입력 데이터에 대한 전처리 및 스케일링을 수행하면 모델의 성능을 더욱 개선할 수 있습니다.

# MODEL
# mlp = MLPRegressor(hidden_layer_sizes=(1024, 512,2),
#                    max_iter=500, # = epochs = 반복 횟수
#                    #activation='logistic',
#                    activation='relu',
#                    solver='adam',
#                    random_state=42,
#                    verbose=1,
#                    #alpha=0.1,
#                    #learning_rate='adaptive',
#                    #early_stopping = False
#                    )
# Iteration 1, loss = 5526.17550062
# Iteration 5, loss = 43.58400418
# Iteration 10, loss = 14.40684563
# Iteration 100, loss = 0.08554883

# MODEL 2
# mlp = MLPRegressor(hidden_layer_sizes=(1024, 512,2),
#                    max_iter=500, # = epochs = 반복 횟수
#                    activation='relu',
#                    solver='lbfgs',
#                    random_state=42,
#                    verbose=1,
#                    #alpha=0.1,
#                    #learning_rate='adaptive',
#                    #early_stopping = False
#                    )
# Valid 데이터 RMSE: 0.393
# activation='logistic', X
# activation='identity', X
# solver='sgd', X

# MODEL 3
# mlp = MLPRegressor(hidden_layer_sizes=(1024, 512,2),
                #    max_iter=500, # = epochs = 반복 횟수
                #    activation='relu',
                #    solver='lbfgs', # 모델 학습에 사용되는 최적화 알고리즘
                #    random_state=42,
                #    verbose=1,
                #    alpha=5,
                #    batch_size=200
                #    #learning_rate='adaptive',
                #    #tol=0.5,
                #    #validation_fraction=0.3,  default = 0.1
                #    #epsilon = 1e-8
                #    )
# Valid 데이터 RMSE: 0.376
# alpha=1, 값을 올리면 과적합 방지해줘서 성능 향상에 도움이 되었다.
# solver='lbfgs', 긍정적.

# MODEL 3
# mlp = MLPRegressor(hidden_layer_sizes=(1024, 512,2),
#                    max_iter=500, # = epochs = 반복 횟수
#                    activation='relu',
#                    solver='lbfgs', # 모델 학습에 사용되는 최적화 알고리즘
#                    random_state=42,
#                    verbose=1,
#                    alpha=50,
#                    batch_size=200
#                    #learning_rate='adaptive',
#                    #tol=0.5,
#                    #validation_fraction=0.3,  default = 0.1
#                    #epsilon = 1e-8
#                    )
# Valid 데이터 RMSE: 0.511

# MODEL 4
# mlp = MLPRegressor(hidden_layer_sizes=(1024, 512,2),
                #    max_iter=1000, # = epochs = 반복 횟수
                #    activation='relu',
                #    solver='lbfgs', # 모델 학습에 사용되는 최적화 알고리즘
                #    random_state=42,
                #    verbose=1,
                #    alpha=5,
                #    #batch_size=200
                #    #learning_rate='adaptive',
                #    #tol=0.5,
                #    #validation_fraction=0.3,  default = 0.1
                #    #epsilon = 1e-8
                #    )
# Valid 데이터 RMSE: 0.409

# MODEL 5
# mlp = MLPRegressor(hidden_layer_sizes=(1024, 512,2),
#                    max_iter=1000, # = epochs = 반복 횟수
#                    activation='relu',
#                    solver='lbfgs', # 모델 학습에 사용되는 최적화 알고리즘
#                    random_state=42,
#                    verbose=1,
#                    alpha=5,
#                    batch_size=200
#                    #learning_rate='adaptive',
#                    #tol=0.5,
#                    #validation_fraction=0.3,  default = 0.1
#                    #epsilon = 1e-8
#                    )
# Valid 데이터 RMSE: 0.409
# batch_size=200 배치는 200이 기본값인거 같고.. 1000번 돌리니 과적합이 생기는건가?

# model 6
# mlp = MLPRegressor(hidden_layer_sizes=(1024, 512,2),
#                    max_iter=500, # = epochs = 반복 횟수
#                    activation='relu',
#                    solver='lbfgs', # 모델 학습에 사용되는 최적화 알고리즘
#                    random_state=42,
#                    verbose=1,
#                    alpha=5,
#                    #batch_size=200,
#                    learning_rate='adaptive',
#                    #tol=0.5,
#                    #validation_fraction=0.3,  default = 0.1
#                    #epsilon = 1e-8
#                    early_stopping=True
#                    )
# early_stopping=True 지표가 안 나오는구나... 500번 다 돈다는건 계속 향상중...?
# learning_rate='adaptive', 실험 결과 = 똑같네?
# Valid 데이터 RMSE: 0.376

# model 7
# mlp = MLPRegressor(hidden_layer_sizes=(1024, 512,2),
#                    max_iter=500, # = epochs = 반복 횟수
#                    activation='relu',
#                    solver='lbfgs', # 모델 학습에 사용되는 최적화 알고리즘
#                    random_state=42,
#                    verbose=1,
#                    alpha=5,
#                    #batch_size=200,
#                    learning_rate='invscaling',
#                    #tol=0.5,
#                    #validation_fraction=0.3,  default = 0.1
#                    #epsilon = 1e-8
#                    early_stopping=True
#                    )
# learning_rate='invscaling', 그대로이다.
# Valid 데이터 RMSE: 0.376

# model 7
# mlp = MLPRegressor(hidden_layer_sizes=(1024, 512,2),
#                    max_iter=500, # = epochs = 반복 횟수
#                    activation='relu',
#                    solver='lbfgs', # 모델 학습에 사용되는 최적화 알고리즘
#                    random_state=42,
#                    verbose=1,
#                    alpha=5,
#                    #batch_size=200,
#                    learning_rate='constant',
#                    #tol=0.5,
#                    #validation_fraction=0.3,  default = 0.1
#                    #epsilon = 1e-8
#                    early_stopping=True,
#                    )
# learning_rate='constant', 실험 들어가즈아~ 도움이 안될 것 같다..
# Valid 데이터 RMSE: 0.376

# model 8
# mlp = MLPRegressor(hidden_layer_sizes=(1024, 512,2),
#                    max_iter=500, # = epochs = 반복 횟수
#                    activation='relu',
#                    solver='lbfgs', # 모델 학습에 사용되는 최적화 알고리즘
#                    random_state=42,
#                    verbose=1,
#                    alpha=5,
#                    #batch_size=200,
#                    learning_rate='constant',
#                    #tol=0.5,
#                    #validation_fraction=0.3,  default = 0.1
#                    #epsilon = 1e-8
#                    early_stopping=True,
#                    learning_rate_init=5
#                    )
# learning_rate_init=5 'contstant'의 짝꿍의 효과는?! 없었다고 한다.
# Valid 데이터 RMSE: 0.376

# model 9
# mlp = MLPRegressor(hidden_layer_sizes=(1024, 512,2),
#                    max_iter=500, # = epochs = 반복 횟수
#                    activation='relu',
#                    solver='lbfgs', # 모델 학습에 사용되는 최적화 알고리즘
#                    random_state=42,
#                    verbose=1,
#                    alpha=5,
#                    #batch_size=200,
#                    #learning_rate='constant',
#                    #tol=0.5,
#                    validation_fraction=0.3, #default = 0.1
#                    #epsilon = 1e-8
#                    early_stopping=True,
#                    #learning_rate_init=5
#                    )
# Valid 데이터 RMSE: 1.992
# validation_fraction=0.3 성능 향상이 보이지 않았다.
# tol 값 수정이 필요해 보인다.
# tol값은 일반적으로 손실 함수 값의 변화량을 나타내며, 이 값이 tol보다 작아지면 최적화 알고리즘이 수렴했다고 판단하고 학습을 종료합니다. 1e-4로 tol값을 설정하면, 손실 함수 값의 변화량이 0.0001보다 작아지면 최적화 알고리즘이 종료되어 학습이 완료됩니다.

# model 10
# mlp = MLPRegressor(hidden_layer_sizes=(1024, 512,2),
#                    max_iter=500, # = epochs = 반복 횟수
#                    activation='relu',
#                    solver='lbfgs', # 모델 학습에 사용되는 최적화 알고리즘
#                    random_state=42,
#                    verbose=1,
#                    alpha=5,
#                    #batch_size=200,
#                    #learning_rate='constant',
#                    tol=1e-6, # 성능향상에는 직접적으로 연관은 없겠구나 
#                    validation_fraction=0.8, #default = 0.1
#                    #epsilon = 1e-8
#                    early_stopping=True,
#                    #learning_rate_init=5
#                    )
# validation_fraction=0.8, 값 그대로이다.

# model 11
# mlp = MLPRegressor(hidden_layer_sizes=(1024, 512,2),
                #    max_iter=500, # = epochs = 반복 횟수
                #    activation='relu',
                #    solver='lbfgs', # 모델 학습에 사용되는 최적화 알고리즘
                #    random_state=42,
                #    verbose=1,
                #    alpha=8,
                #    tol=1e-6, # 성능향상에는 직접적으로 연관은 없겠구나 
                #    #epsilon = 1e-8
                #    early_stopping=True,
                #    shuffle=True
                #    )
# Valid 데이터 RMSE: 0.451

# model 11
# mlp = MLPRegressor(hidden_layer_sizes=(1024, 512,2),
#                    max_iter=500, # = epochs = 반복 횟수
#                    activation='relu',
#                    solver='lbfgs', # 모델 학습에 사용되는 최적화 알고리즘
#                    random_state=42,
#                    verbose=1,
#                    alpha=3,
#                    tol=1e-6, # 성능향상에는 직접적으로 연관은 없겠구나 
#                    #epsilon = 1e-8
#                    early_stopping=True,
#                    shuffle=True
#                    )
# Valid 데이터 RMSE: 0.371

# # model 11
# mlp = MLPRegressor(hidden_layer_sizes=(512, 256,3),
#                    max_iter=500, # = epochs = 반복 횟수
#                    activation='relu',
#                    solver='lbfgs', # 모델 학습에 사용되는 최적화 알고리즘
#                    random_state=42,
#                    verbose=1,
#                    alpha=3,
#                    tol=1e-6, # 성능향상에는 직접적으로 연관은 없겠구나 
#                    #epsilon = 1e-8
#                    early_stopping=True,
#                    shuffle=True
#                    )
# # Valid 데이터 RMSE: 0.366

# model 11
# mlp = MLPRegressor(hidden_layer_sizes=(1024, 512,3),
#                    max_iter=500, # = epochs = 반복 횟수
#                    activation='relu',
#                    solver='lbfgs', # 모델 학습에 사용되는 최적화 알고리즘
#                    random_state=42,
#                    verbose=1,
#                    alpha=3,
#                    tol=1e-6, # 성능향상에는 직접적으로 연관은 없겠구나 
#                    #epsilon = 1e-8
#                    early_stopping=True,
#                    shuffle=True
#                    )
# # Valid 데이터 RMSE: 0.844

# # model 11
# mlp = MLPRegressor(hidden_layer_sizes=(512, 256,3),
#                    max_iter=500, # = epochs = 반복 횟수
#                    activation='relu',
#                    solver='lbfgs', # 모델 학습에 사용되는 최적화 알고리즘
#                    random_state=42,
#                    verbose=1,
#                    alpha=3,
#                    tol=1e-6, # 성능향상에는 직접적으로 연관은 없겠구나 
#                    #epsilon = 1e-8
#                    early_stopping=True,
#                    shuffle=True
#                    )
# # Valid 데이터 RMSE: 0.366

# # model 12
# mlp = MLPRegressor(hidden_layer_sizes=(512, 256,3),
#                    max_iter=500, # = epochs = 반복 횟수
#                    activation='relu',
#                    solver='lbfgs', # 모델 학습에 사용되는 최적화 알고리즘
#                    random_state=42,
#                    verbose=1,
#                    alpha=3.5,
#                    tol=1e-6, # 성능향상에는 직접적으로 연관은 없겠구나 
#                    #epsilon = 1e-8
#                    early_stopping=True,
#                    shuffle=True
#                    )
# # Valid 데이터 RMSE: 0.365

# model 13
mlp = MLPRegressor(hidden_layer_sizes=(512, 256,3),
                   max_iter=500, # = epochs = 반복 횟수
                   activation='relu',
                   solver='lbfgs', # 모델 학습에 사용되는 최적화 알고리즘
                   random_state=42,
                   verbose=1,
                   alpha=3.5,
                   #tol=1e-6, # 성능향상에는 직접적으로 연관은 없겠구나 
                   #epsilon = 1e-8
                   #early_stopping=True,
                   #shuffle=True
                   )
# Valid 데이터 RMSE: 0.365




#  "f"는 현재까지 계산된 손실 함수(Loss Function)의 값으로, 모델이 학습 중에 얼마나 잘 작동하고 있는지를 나타냅니다.
# 손실 함수 값이 감소할수록 모델의 성능이 향상되고 있다는 의미입니다.

# "|proj g|"는 현재까지 계산된 경사도(Gradient)의 크기를 나타내며, 이 값이 작을수록 모델이 최적점에 접근하고 있다는 의미입니다. 경사하강법은 손실 함수의 기울기(Gradient)를 사용하여 최적점을 찾기 때문에, 경사도 값이 작아질수록 최적점에 가까워지고 있는 것입니다.






################################print(mlp.get_params())#############
# (hidden_layer_sizes: ArrayLike | tuple[int, int] = ..., 
# activation: Literal['relu', 'identity', 'logistic', 'tanh'] = "relu", *,
# solver: Literal['lbfgs', 'sgd', 'adam'] = "adam",
# alpha: Float = 0.0001,
# batch_size: Int | str = "auto",
# learning_rate: Literal['constant', 'invscaling', 'adaptive'] = "constant",
# learning_rate_init: Float = 0.001,
# power_t: Float = 0.5,
# max_iter: Int = 200,
# shuffle: bool = True,
# random_state: Int | RandomState | None = None,
# tol: Float = 0.0001,
# verbose: bool = False,
# warm_start: bool = False,
# momentum: Float = 0.9,
# nesterovs_momentum: bool = True,
# early_stopping: bool = False,
# validation_fraction: Float = 0.1,
# beta_1: Float = 0.9,
# beta_2: Float = 0.999,
# epsilon: Float = 1e-8,
# n_iter_no_change:
# Int = 10,
# max_fun: Int = 15000) -> None

mlp.fit(X_train, y_train)

# valid PREDICT
y_pred_valid = mlp.predict(X_valid)
rmse_valid = np.sqrt(mean_squared_error(y_valid, y_pred_valid))
print(f"Valid 데이터 RMSE: {rmse_valid:.3f}")

# test PREDICT
X_test = test_df.values
X_poly_test = poly.transform(X_test)
X_test_scaled = scaler.transform(X_poly_test)
y_pred_test = mlp.predict(X_test_scaled)

date = datetime.datetime.now()
date = date.strftime("%m%d-%H%M")
start_time = time.time()

# SUBMIT
sample_submission_df['Calories_Burned'] = y_pred_test
sample_submission_df.to_csv(save_path + date + 'submission_MLP_Poly.csv', index=False)

# hidden_layer_sizes: 각 숨겨진 레이어의 뉴런 수를 지정하는 튜플입니다.
# 예를 들어 이면 hidden_layer_sizes=(10, 5)모델에는 각각 10개와 5개의 뉴런이 있는 두 개의 숨겨진 레이어가 있습니다.

# activation: 은닉층에서 사용되는 활성화 함수.
# 가능한 값은 'identity', 'logistic', 'tanh'및 입니다 'relu'.

# solver: 모델 학습에 사용되는 최적화 알고리즘입니다.
# 가능한 값은 'lbfgs', 'sgd'및 입니다 'adam'.

# alpha: L2 정규화 매개변수입니다.
# 알파 값이 높을수록 정규화가 더 많아져 과적합을 방지할 수 있습니다.

# batch_size: 확률적 경사하강법을 위해 각 배치에서 사용되는 샘플 수입니다.
# 가능한 값은 배치 크기 또는 정수 값으로 'auto'사용하는 입니다.min(200, n_samples)

# learning_rate: 가중치 업데이트를 위한 학습률 일정입니다.
# 가능한 값은 'constant', 'invscaling'및 입니다 'adaptive'.

# learning_rate_init: 사용된 초기 학습률입니다.
# learning_rate이 매개변수는 가 로 설정된 경우에만 사용됩니다 'constant'.

# max_iter: 교육 중에 수행할 최대 반복 횟수입니다.

# shuffle: 각 에포크 전에 훈련 데이터를 섞을지 여부.

# random_state: 난수 생성기에서 사용하는 시드입니다.

# tol: 최적화 알고리즘의 수렴에 대한 허용오차.

# verbose: 훈련 중 진행 메시지를 인쇄할지 여부.

# momentum: 경사하강법 업데이트를 위한 추진력. 경우에만 사용됩니다 solver='sgd'.

# nesterovs_momentum: Nesterov의 모멘텀 사용 여부. solver='sgd'및 에만 사용됩니다 momentum > 0.

# early_stopping: 유효성 검사 점수가 향상되지 않을 때 교육을 종료하기 위해 조기 중지를 사용할지 여부입니다.

# validation_fraction: 검증 데이터로 사용할 훈련 데이터의 비율입니다.

# beta_1: adam 옵티마이저에서 첫 번째 모멘트 벡터 추정에 대한 지수적 감쇠율입니다.

# beta_2: 아담 옵티마이저에서 2차 모멘트 벡터 추정에 대한 지수적 감쇠율.

# epsilon: adam 옵티마이저에서 0으로 나누기를 피하기 위한 값.

# 이러한 매개변수 중 일부는 특정 최적화 알고리즘이나 학습률 일정에만 적용되므로 자세한 내용은 scikit-learn 문서를 참조하는 것이 중요합니다. 또한 이러한 매개변수의 최적 값은 해결하려는 특정 문제에 따라 달라질 수 있으며 그리드 검색 또는 무작위 검색과 같은 기술을 사용하여 조정해야 합니다.