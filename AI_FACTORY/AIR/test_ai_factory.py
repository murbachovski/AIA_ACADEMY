from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd
import datetime
from sklearn.model_selection import train_test_split
# Load train and test data
path='./_data/ai_factory/'
save_path= './_save/ai_factory/'
train_data = pd.read_csv(path+'train_data.csv')
test_data = pd.read_csv(path+'test_data.csv')
submission = pd.read_csv(path+'answer_sample.csv')

# 
features = ['air_inflow', 'air_end_temp', 'out_pressure', 'motor_current', 'motor_rpm', 'motor_temp', 'motor_vibe']

# Prepare train and test data
all_data = test_data[features]

# 
X_train, X_test = train_test_split(all_data, test_size= 0.9, random_state= 337)
print(X_train.shape, X_test.shape)

scaler = StandardScaler()
X_test = scaler.fit_transform(X_test)

pca = PCA(n_components=4, random_state=337)
X_test = pca.fit_transform(X_test)
print(X_test)
print(X_test.shape)
from sklearn.metrics import silhouette_samples, silhouette_score

# sc_max = 0
# for i in range(200, 210):
#     for j in range(200, 202):    
#         dbscan3 = DBSCAN(eps=0.1 * i, min_samples=j, metric='euclidean')
#         aaa = dbscan3.fit_predict(all_tf)
        
#         sc = silhouette_score(all_tf, aaa)
#         print("실루엣 스코어 : ", sc)
        
#         print('스캔 : ', i, j, ":", 
#             np.unique(dbscan3.labels_, return_counts=True)) 

#         if sc > sc_max:
#             sc_max = sc
#             best_parameters = {'eps' : i, 'min_samples' : j}

# print("최고 실루엣 : ", sc_max)
# print("최적의 매개변수 : ", best_parameters)
# # 최고 실루엣 :  0.7814148871578233
# # 최적의 매개변수 :  {'eps': 7, 'min_samples': 3}


dbscan3 = DBSCAN(eps= 15, min_samples= 50, metric='euclidean')
# 0.08 / 15 = 350
y_pred_test_lof = dbscan3.fit_predict(X_test)
print(np.unique(y_pred_test_lof, return_counts=True))       # (array([0, 1], dtype=int64), array([9836,   16], dtype=int64))
print(np.unique(y_pred_test_lof[2463:], return_counts=True))
print(y_pred_test_lof.shape)

lof_predictions = [1 if x == -1 else 0 for x in y_pred_test_lof]
#lof_predictions = [0 if x == -1 else 0 for x in y_pred_test_lof]

submission['label'] = pd.DataFrame({'Prediction': lof_predictions})
print(submission.value_counts())
print(submission['label'].value_counts())

#time
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

submission.to_csv(save_path + date + '_REAL_DB_submission.csv', index=False)
print(submission['label'].shape)

# eps: 각 포인트 주변의 이웃 반경. 이 반경 내의 점은 이웃으로 간주됩니다.
# min_samples: 조밀한 영역을 형성하는 데 필요한 최소 포인트 수. 이웃이 충분하지 않은 포인트는 노이즈로 간주됩니다.
# 메트릭: 포인트 간의 거리를 계산하는 데 사용되는 거리 메트릭입니다. 이때 유클리드 거리를 사용한다.