# 그물망처럼 찾겠다.
# 파라미터 전체를 다 하겠다. / 모델 정의 부분이나 모델 훈련 부분에 있음

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold,cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

# 1. 데이터
x,y = load_iris(return_X_y=True)

x_train,x_test,y_train,y_test = train_test_split(
    x,y, shuffle=True, random_state=337, test_size=0.2, stratify=y # stratify=y 사용함으로써 각 라벨값이 골고루 분배된다 
)    

# SVC 모델에 파라미터
gamma = [0.001,0.01,0.1,1,10,100] 
C = [0.001,0.01,0.1,1,10,100] # 곡선으로 휜다.


max_score=0

for i in gamma:
    for j in C:
#2.모델
        model = SVC(gamma=i, C=j)
#3. 컴파일 훈련
        model.fit(x_train,y_train)
#4. 평가 예측 # keras _ evaluate -> sklearn _ score
        score = model.score(x_test,y_test)
        print('acc : ', score)
        
        if max_score < score:
            max_score = score
            best_parameters = {'gamma':i,'C':j} # score가 바뀌어야 바뀌니까 최고의 스코어에 최고의 파라이터 일때만 갱신됨
            
print('최고점수 :', max_score)
print('최적의 매개변수 :', best_parameters) # 매개변수 = 파라미터

# acc :  0.9666666666666667