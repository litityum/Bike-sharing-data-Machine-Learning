# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 23:00:38 2020

@author: vamsi
"""

import pandas as pd
import numpy as np


# %% For the third session :  Data visualization 
from matplotlib.pyplot import figure, subplot, hist, xlabel,show
from matplotlib.pyplot import boxplot, xticks, ylabel, title
from matplotlib.pyplot import plot, yticks,legend 
from scipy import stats
from scipy.linalg import svd
import matplotlib.pyplot as plt

X = pd.read_csv ('london_merged.csv')
N = len(X)

date= [[]] *N
time =[[]] *N
for i in range(N):
    date[i],time[i] = str(X['timestamp'][i]).split( )   
    time[i]=int(time[i][0]+time[i][1])
  
#merge t1&t2 into on column t
t=(X.t1+X.t2)/2

#create new dataframe with manipulated data
X_new = pd.DataFrame(np.array([time,t,X['hum'],X['wind_speed']]).T,
                     columns=['time','temperature','hum','wind_speed'])
# do 1 out of k in weather_code
weather_code = np.array(X['weather_code'], dtype=int).T
code_diff = [1,2,3,4,7,10,26] # no 94
K = len(code_diff)
weather_code_encoding = np.zeros((weather_code.size, K))
for i in range(weather_code.size):
    weather_code_encoding[i, code_diff.index(weather_code[i])] = 1

X_new=X_new.join(pd.DataFrame(weather_code_encoding,
                              columns=['Weather_Clear','Weather_ scattered clouds',
                                       'Weather_ Broken clouds ','Weather_ Cloudy',
                                       'Weather_ Rain','Weather_ rain With thunderstorm',
                                       'Weather_ snoWfall']))
# add original columns to new dataframe
X_new=X_new.join(pd.DataFrame(X[X.columns[[7,8]]]))

# Since season is a categorical variable,do a one-out-of-K encoding of the variable:
season = np.array(X['season'], dtype=int).T
K = season.max()+1
season_encoding = np.zeros((season.size, K))
season_encoding[np.arange(season.size), season] = 1

X_new = X_new.join(pd.DataFrame(season_encoding,
                   columns=['winter','spring','summer','autumn']))




Y = X.cnt
X = X_new.to_numpy()
y = Y.to_numpy()
attributeNames = list(X_new.columns)

#Baseline for the model 
max_weather= np.bincount(season).argmax()
if max_weather == 0:
  print("Spring")
elif max_weather == 1:
  print("Summer")  
elif max_weather == 2:
  print("Fall")  
elif max_weather == 3:
  print("Winter")
  
  
  