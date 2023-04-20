from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd
import datetime

# Load train and test data
path='./_data/ai_factory/'
save_path= './_save/ai_factory/'
train_data = pd.read_csv(path+'train_data.csv')
test_data = pd.read_csv(path+'test_data.csv')
submission = pd.read_csv(path+'answer_sample.csv')

# 
features = ['air_inflow', 'air_end_temp', 'out_pressure', 'motor_current', 'motor_rpm', 'motor_temp', 'motor_vibe']

# Prepare train and test data
all_data = train_data[features]

scaler = StandardScaler()
all_data = scaler.fit_transform(all_data)

pca = PCA(n_components=7, random_state=10)
all_tf = pca.fit_transform(all_data)
print(all_tf)
print(all_data.shape)
from sklearn.metrics import silhouette_samples, silhouette_score

sc_max = 0
for i in range(3, 5):
    for j in range(3, 5):    
        dbscan3 = DBSCAN(eps=0.1 * i, min_samples=j, metric='euclidean')
        aaa = dbscan3.fit_predict(all_tf)
        
        sc = silhouette_score(all_tf, aaa)
        print("실루엣 스코어 : ", sc)
        
        print('스캔 : ', i, j, ":", 
            np.unique(dbscan3.labels_, return_counts=True)) 

        if sc > sc_max:
            sc_max = sc
            best_parameters = {'eps' : i, 'min_samples' : j}

print("최고 실루엣 : ", sc_max)
print("최적의 매개변수 : ", best_parameters)
# 최고 실루엣 :  0.7814148871578233
# 최적의 매개변수 :  {'eps': 7, 'min_samples': 3}


dbscan5 = DBSCAN(eps= 0.9, min_samples= 10, metric='euclidean')
y_pred_test_lof = dbscan5.fit_predict(all_tf)
print(np.unique(y_pred_test_lof, return_counts=True))       # (array([0, 1], dtype=int64), array([9836,   16], dtype=int64))
print(np.unique(y_pred_test_lof[2463:], return_counts=True))
print(y_pred_test_lof.shape)

lof_predictions = [1 if x == -1 else 0 for x in y_pred_test_lof]
#lof_predictions = [0 if x == -1 else 0 for x in y_pred_test_lof]

submission['label'] = pd.DataFrame({'Prediction': lof_predictions})
print(submission.value_counts())

#time
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

submission.to_csv(save_path + date + '_REAL_DB_submission.csv', index=False)
print(submission['label'].shape)
