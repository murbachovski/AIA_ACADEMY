import pandas as pd
import os
import numpy as np
from typing import List,Tuple

def file_path()->str:
    return '_data/finedust'

def sort_data_by_first_element(*args: List[List]) -> Tuple[List]:
    return tuple(map(list, zip(*sorted(zip(*args), key=lambda x: x[0]))))

def create_awsmap_pmmap() -> Tuple[pd.DataFrame, pd.DataFrame]:
    path = file_path()
    path_list = os.listdir(path)
    meta = '/'.join([path, path_list[1]])
    meta_list = os.listdir(meta)
    file_name_aws = meta_list[0]
    awsmap = pd.read_csv('/'.join([meta, file_name_aws]))
    awsmap = awsmap.drop(awsmap.columns[-1], axis=1)
    file_name_pm = meta_list[1]
    pmmap = pd.read_csv('/'.join([meta, file_name_pm]))
    pmmap = pmmap.drop(pmmap.columns[-1], axis=1)
    return awsmap, pmmap

def create_train(filename:str)->pd.DataFrame:
    path=file_path()
    path_list=os.listdir(path)    
    train='/'.join([path,path_list[4]])
    return pd.read_csv(f'{train}/{filename}.csv')

def create_train_AWS(filename:str)->pd.DataFrame:
    path=file_path()
    path_list=os.listdir(path)  
    train_aws='/'.join([path,path_list[5]])
    return pd.read_csv(f'{train_aws}/{filename}.csv')

def create_test_input(filename:str)->pd.DataFrame:
    path=file_path()
    path_list=os.listdir(path)    
    test_input='/'.join([path,path_list[3]])
    return pd.read_csv(f'{test_input}/{filename}.csv')

def create_test_AWS(filename:str)->pd.DataFrame:
    path=file_path()
    path_list=os.listdir(path)  
    test_aws='/'.join([path,path_list[2]])
    return pd.read_csv(f'{test_aws}/{filename}.csv')

############################################거리자료############################################
def all_distance_info() -> pd.DataFrame:
    awsmap, pmmap = create_awsmap_pmmap()
    awslocation=[]
    pmlocation=[]
    Distance=[]
    for i in range(awsmap.shape[0]):
        for j in range(pmmap.shape[0]):
            awslocation.append(awsmap["Location"][i])
            pmlocation.append(pmmap["Location"][j])
            LatitudeDis=awsmap["Latitude"][i]-pmmap["Latitude"][j]
            LongitudeDis=awsmap["Longitude"][i]-pmmap["Longitude"][j]
            Distance.append(np.sqrt(LatitudeDis**2+LongitudeDis**2))
    pmlocation,awslocation,Distance=sort_data_by_first_element(pmlocation,awslocation,Distance)
    awslocation=pd.Series(awslocation,name='awslocation')
    pmlocation=pd.Series(pmlocation,name='pmlocation')
    Distance=pd.Series(Distance,name='Distance')
    LocationInfo=pd.concat([awslocation,pmlocation,Distance],axis=1)
    return LocationInfo

##########################################최인접 자료#########################################

def min_distance_info()->pd.DataFrame:
    awsmap, pmmap = create_awsmap_pmmap()
    awslocation=[]
    pmlocation=[]
    Distance=[]
    for j in range(pmmap.shape[0]):
        min_distance=np.sqrt(2)*180
        min_awd=''
        pmlocation.append(pmmap["Location"][j])
        for i in range(awsmap.shape[0]):
            LatitudeDis=awsmap["Latitude"][i]-pmmap["Latitude"][j]
            LongitudeDis=awsmap["Longitude"][i]-pmmap["Longitude"][j]
            current_dis=np.sqrt(LatitudeDis**2+LongitudeDis**2)
            if current_dis<min_distance:
                min_awd=awsmap["Location"][i]
                min_distance=current_dis
        awslocation.append(min_awd)
        Distance.append(min_distance)
    pmlocation,awslocation,Distance=sort_data_by_first_element(pmlocation,awslocation,Distance)
    awslocation=pd.Series(awslocation,name='awslocation')
    pmlocation=pd.Series(pmlocation,name='pmlocation')
    Distance=pd.Series(Distance,name='Distance')
    LocationInfo=pd.concat([pmlocation,awslocation,Distance],axis=1)
    LocationInfo
    return LocationInfo



def load_min_distance(print_download:bool=True,load_name:bool=True
                      )->Tuple[List[pd.DataFrame],List[pd.DataFrame],List[str]]:
    '''각 PM측정소의 최근접 AWS측정소 데이터와 PM측정소 데이터를 concate해서 출력해줌
    
    output: train_datas,test_datas,pmlocation
    '''
    distance_info=min_distance_info()
    awslocation=distance_info['awslocation']
    pmlocation=distance_info['pmlocation']
    Distance=distance_info['Distance']
    train_data_for_pm=[]
    test_data_for_pm=[]
    for i in range(len(pmlocation)):
        pm_train=create_train(pmlocation[i])
        pm_test=create_test_input(pmlocation[i])
        aws_train=create_train_AWS(awslocation[i])
        aws_test=create_test_AWS(awslocation[i])
        if print_download:
            print(f'{pmlocation[i]}파일 로드 완료')
        train_data_for_pm.append(pd.concat([aws_train.drop(['지점'],axis=1),pm_train['PM2.5']],axis=1))
        test_data_for_pm.append(pd.concat([aws_test.drop(['지점'],axis=1),pm_test['PM2.5']],axis=1))
    if load_name:
        return train_data_for_pm,test_data_for_pm,pmlocation
    else:
        return train_data_for_pm,test_data_for_pm

def min_distance_info_sample()->pd.DataFrame:
    awsmap, pmmap = create_awsmap_pmmap()
    awslocation=[]
    pmlocation=[]
    Distance=[]
    min_distance=np.sqrt(2)*180
    min_awd=''
    for j in range(2):
        pmlocation.append(pmmap["Location"][j])
        for i in range(awsmap.shape[0]):
            LatitudeDis=awsmap["Latitude"][i]-pmmap["Latitude"][j]
            LongitudeDis=awsmap["Longitude"][i]-pmmap["Longitude"][j]
            current_dis=np.sqrt(LatitudeDis**2+LongitudeDis**2)
            if current_dis<min_distance:
                min_awd=awsmap["Location"][i]
                min_distance=current_dis
        awslocation.append(min_awd)
        Distance.append(min_distance)
    pmlocation,awslocation,Distance=sort_data_by_first_element(pmlocation,awslocation,Distance)
    awslocation=pd.Series(awslocation,name='awslocation')
    pmlocation=pd.Series(pmlocation,name='pmlocation')
    Distance=pd.Series(Distance,name='Distance')
    LocationInfo=pd.concat([pmlocation,awslocation,Distance],axis=1)
    LocationInfo
    return LocationInfo


def load_min_distance_sample(print_download:bool=True,load_name:bool=True
                      )->Tuple[List[pd.DataFrame],List[pd.DataFrame],List[str]]:
    '''각 PM측정소의 최근접 AWS측정소 데이터와 PM측정소 데이터를 concate해서 출력해줌'''
    distance_info=min_distance_info_sample()
    awslocation=distance_info['awslocation']
    pmlocation=distance_info['pmlocation']
    Distance=distance_info['Distance']
    train_data_for_pm=[]
    test_data_for_pm=[]
    for i in range(len(pmlocation)):
        pm_train=create_train(pmlocation[i])
        pm_test=create_test_input(pmlocation[i])
        aws_train=create_train_AWS(awslocation[i])
        aws_test=create_test_AWS(awslocation[i])
        if print_download:
            print(f'{pmlocation[i]}파일 로드 완료')
        train_data_for_pm.append(pd.concat([aws_train.drop(['지점'],axis=1),pm_train['PM2.5']],axis=1))
        test_data_for_pm.append(pd.concat([aws_test.drop(['지점'],axis=1),pm_test['PM2.5']],axis=1))
    if load_name:
        return train_data_for_pm,test_data_for_pm,pmlocation
    else:
        return train_data_for_pm,test_data_for_pm
