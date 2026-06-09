import pandas as pd
import numpy as np
delivery=pd.read_csv('deliveries.csv.zip')
matches=pd.read_csv("matches.csv")
print(delivery.head())
print(matches.head())
print(delivery.shape)
print(matches.shape)
total_score_df=delivery.groupby(['match_id','inning']).sum()['total_runs'].reset_index()

total_score_df=total_score_df[total_score_df['inning']==1]
print(total_score_df)
match_df=matches.merge(total_score_df[['match_id','total_runs']],left_on='id',right_on='match_id')
print(match_df)
teams = [
    'Sunrisers Hyderabad',
    'Mumbai Indians',
    'Royal Challengers Bangalore',
    'Kolkata Knight Riders',
    'Kings XI Punjab',
    'Chennai Super Kings',
    'Rajasthan Royals',
    'Delhi Capitals'
]
match_df['team1']=match_df['team1'].str.replace('Delhi Daredevils','Delhi Capitals')
match_df['team2']=match_df['team2'].str.replace('Delhi Daredevils','Delhi Capitals')

match_df['team1']=match_df['team1'].str.replace('Deccan Chargers','Sunrisers Hyderabad')
match_df['team2']=match_df['team2'].str.replace("Deccan Chargers",'Sunrisers Hyderabad')
match_df=match_df[match_df['team1'].isin(teams)]
match_df=match_df[match_df['team2'].isin(teams)]
match_df.shape
# print(match_df)
match_df=match_df[match_df['dl_applied']==0]
print(match_df)
match_df=match_df[["match_id",'city','winner','total_runs']]
delivery_df=match_df.merge(delivery,on='match_id')

delivery_df=delivery_df[delivery_df['inning']==2]

delivery_df['curr_score'] = delivery_df.groupby('match_id')['total_runs_y'].cumsum()
delivery_df['runs_left']=delivery_df['total_runs_x']-delivery_df['curr_score']
print(delivery_df)
delivery_df['balls_left']=126-(delivery_df['over']*6+delivery_df['ball'])
print(delivery_df)
delivery_df['player_dismissed']=delivery_df['player_dismissed'].fillna('0')
delivery_df['player_dismissed']=delivery_df['player_dismissed'].apply(lambda x :x if x=='0' else '1')
delivery_df['player_dismissed']=delivery_df['player_dismissed'].astype(int)
wickets=delivery_df.groupby('match_id')['player_dismissed'].cumsum().values
delivery_df['wickets']=10-wickets
print(delivery_df.head())
print(delivery_df.tail())
delivery_df['curr_runrate']=delivery_df['curr_score']*6/(120-delivery_df['balls_left'])
delivery_df["req_runrate"]=delivery_df['runs_left']*6/delivery_df['balls_left']
print(delivery_df.sample(7))
def result(row):
    return 1 if row['winner']==row['batting_team'] else 0

delivery_df['result']=delivery_df.apply(result,axis=1)
print(delivery_df.columns)
final_df=delivery_df[['batting_team', 'bowling_team', 'city','runs_left','balls_left','wickets','total_runs_x','curr_runrate','req_runrate','result']]
final_df=final_df.sample(final_df.shape[0])
final_df=final_df.dropna()
final_df=final_df[final_df['balls_left']!=0]
print(final_df.sample())
from sklearn.model_selection import train_test_split
X=final_df.drop(columns="result")
y=final_df['result']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=44)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
trf=ColumnTransformer([(
    'trf',OneHotEncoder(sparse_output=False,drop='first'),['batting_team','bowling_team','city']
)

],remainder='passthrough')
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
pipe=Pipeline(steps=[
    ('step1',trf),
    ('step2',LogisticRegression(solver='liblinear'))
])
pipe.fit(X_train,y_train)
y_pred=pipe.predict(X_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))
import pickle
pickle.dump(pipe,open("pipe.pkl",'wb'))
print(delivery_df['city'].unique())
