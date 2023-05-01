import pandas as pd
def split_month_day_hour(DataFrame:pd.DataFrame)->pd.DataFrame:
    month_date_time_min=[i.split(' ') for i in DataFrame['일시']]
    DataFrame=DataFrame.drop(['연도','일시'],axis=1)
    month_date=[j.split('-')for j in [i[0] for i in month_date_time_min]]
    time_min=[j.split(':')for j in[i[1] for i in month_date_time_min]]
    month=pd.Series([float(i[0]) for i in month_date],name='월')
    date=pd.Series([float(i[1]) for i in month_date],name='일')
    time=pd.Series([float(i[0])for i in time_min],name='시')
    DataFrame=pd.concat([month,date,time,DataFrame],axis=1)
    return DataFrame

combined_df = pd.read_csv('./_data/ai_factory/social/train_aws_all.csv')

combined_df = split_month_day_hour(combined_df)

combined_df.to_csv('./_data/ai_factory/social/train_aws_all.csv', index=False)

