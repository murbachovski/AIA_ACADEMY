import os
from read_file import load_min_distance
from preprocessing import split_month_day_hour
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from xgboost import XGBRegressor
def save_concated_data_min_distance(print_download:bool=True)->None:
    fortrain,fortest,pmname=load_min_distance(print_download=print_download,load_name=True)
    imputer=IterativeImputer(XGBRegressor(tree_method='gpu_hist',
                       predictor='gpu_predictor',
                       gpu_id=0,n_estimators=50,learning_rate=0.5,
                       max_depth=9))    
    for i in range(len(fortrain)):
        fortrain[i]=split_month_day_hour(fortrain[i])
        fortest[i]=split_month_day_hour(fortest[i])
        imputer.fit_transform(fortrain[i])
        fortrain[i]=fortrain[i].fillna(method='ffill').fillna(method='bfill')
        os.makedirs('./03.AI_finedust/for_Train', exist_ok=True)
        os.makedirs(f'./03.AI_finedust/for_Train/test/', exist_ok=True)
        os.makedirs(f'./03.AI_finedust/for_Train/train/', exist_ok=True)  
        fortrain[i].to_csv(f'./03.AI_finedust/for_Train/train/{pmname[i]}.csv',index=False)
        fortest[i].to_csv(f'./03.AI_finedust/for_Train/test/{pmname[i]}.csv',index=False)
        if print_download:
            print(f'{pmname[i]}저장 완료!')
save_concated_data_min_distance()