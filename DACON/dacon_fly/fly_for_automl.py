import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, MaxAbsScaler, Normalizer, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import PolynomialFeatures
import optuna
import datetime
import warnings
warnings.filterwarnings('ignore')
import random
import os
import numpy as np
import pandas as pd
import gc
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, make_scorer
from xgboost import XGBClassifier


path_save = './_save/_dacon/dacon_fly/'
def RMSE(x, y):
    return np.sqrt(mean_squared_error(x, y))

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(42) # Fixed Seed

def csv_to_parquet(csv_path, save_name):
    df = pd.read_csv(csv_path)
    df.to_parquet(f'./{save_name}.parquet')
    del df
    gc.collect()
    print(save_name, 'Done.')

csv_to_parquet('./_data/dacon/dacon_fly/train.csv', 'train')
csv_to_parquet('./_data/dacon/dacon_fly/test.csv', 'test')

train = pd.read_parquet('./train.parquet')
test = pd.read_parquet('./test.parquet')
submit_csv = pd.read_csv('./_data/dacon/dacon_fly/sample_submission.csv', index_col = 0)

# Replace variables with missing values except for the label (Delay) with the most frequent values of the training data
# 컬럼의 누락된 값은 훈련 데이터에서 해당 컬럼의 최빈값으로 대체됩니다.
NaN_col = ['Origin_State','Destination_State','Airline','Estimated_Departure_Time', 'Estimated_Arrival_Time','Carrier_Code(IATA)','Carrier_ID(DOT)']

for col in NaN_col:
    mode = train[col].mode()[0]
    train[col] = train[col].fillna(mode)
    
    if col in test.columns:
        test[col] = test[col].fillna(mode)
print('Done.')

# Quantify qualitative variables
# 정성적 변수는 LabelEncoder를 사용하여 숫자로 인코딩됩니다.
qual_col = ['Origin_Airport', 'Origin_State', 'Destination_Airport', 'Destination_State', 'Airline', 'Carrier_Code(IATA)', 'Tail_Number']

for i in qual_col:
    le = LabelEncoder()
    le = le.fit(train[i])
    train[i] = le.transform(train[i])
    
    for label in np.unique(test[i]):
        if label not in le.classes_:
            le.classes_ = np.append(le.classes_, label)
    test[i] = le.transform(test[i])
print('Done.')

# Remove unlabeled data
# 훈련 세트에서 레이블이 지정되지 않은 데이터가 제거되고 숫자 레이블 열이 추가됩니다.
train = train.dropna()

column_number = {}
for i, column in enumerate(submit_csv.columns):
    column_number[column] = i
    
def to_number(x, dic):
    return dic[x]

train.loc[:, 'Delay_num'] = train['Delay'].apply(lambda x: to_number(x, column_number))
print('Done.')

train_x = train.drop(columns=['ID', 'Delay', 'Delay_num'])
train_y = train['Delay_num']
test_x = test.drop(columns=['ID'])

# 교육 데이터는 교육 및 검증 세트로 분할되고 수치 기능은 StandardScaler를 사용하여 정규화됩니다.
# 모델은 GridSearchCV와 5겹 교차 검증을 사용하여 수행되는 하이퍼파라미터 튜닝과 함께 XGBClassifier를 사용하여 훈련됩니다.
# Split the training dataset into a training set and a validation set
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2, random_state=42)

# Normalize numerical features
scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
val_x = scaler.transform(val_x)
test_x = scaler.transform(test_x)

validation_indices = np.load('./AutoML_1/split_validation_indices.npy')
# train_indicdes = np.load('AutoML_04/split_train_indices.npy')
# x = x.iloc[train_indicdes]
# y = y.iloc[train_indicdes]

for i in range(10000):
    kf = KFold(n_splits=5, shuffle=True, random_state=i)
    val_x = val_x[validation_indices]
    val_y = val_y[validation_indices]

    def objective(trial):
        alpha = trial.suggest_loguniform('alpha', 0.0000001, 0.1)
        n_restarts_optimizer  = trial.suggest_int('n_restarts_optimizer', 1, 20)
        optimizer = trial.suggest_categorical('optimizer', ['fmin_l_bfgs_b'])

        model = GaussianProcessRegressor(
            alpha=alpha,
            n_restarts_optimizer=n_restarts_optimizer,
            optimizer=optimizer,
        )
        
        model.fit(train_x, train_y)
        
        print('GPR result : ', model.score(val_x, train_y))
        
        y_pred = np.round(model.predict(val_x))
        
        rmse = RMSE(val_y, y_pred)
        print('GPR RMSE : ', rmse)
        if rmse < 0.16:
            submit_csv['Delay_num'] = np.round(model.predict(test_x))
            date = datetime.datetime.now()
            date = date.strftime('%m%d_%H%M%S')
            submit_csv.to_csv(path_save + date + str(round(rmse, 5)) + '.csv')
        return rmse
    opt = optuna.create_study(direction='minimize')
    opt.optimize(objective, n_trials=20)
    print('best param : ', opt.best_params, 'best rmse : ', opt.best_value)