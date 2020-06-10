#!/usr/bin/env python
# coding: utf-8

# %% For the first session : A description of your data set
import pandas as pd
import numpy as np

# ### Attributes
# 
# * "timestamp" - timestamp field for grouping the data
# * "cnt" - the count of a new bike shares
# * "t1" - real temperature in C
# * "t2" - temperature in C "feels like"
# * "hum" - humidity in percentage
# * "windspeed" - wind speed in km/h
# * "weathercode" - category of the weather
# * "isholiday" - boolean field - 1 holiday / 0 non holiday
# * "isweekend" - boolean field - 1 if the day is weekend
# * "season" - category field meteorological seasons: 0-spring ; 1-summer; 2-fall; 3-winter.
# * "weathe_code" category description:
# 1 = Clear ; mostly clear but have some values with haze/fog/patches of fog/ fog in vicinity 
# 2 = scattered clouds / few clouds 
# 3 = Broken clouds 
# 4 = Cloudy 
# 7 = Rain/ light Rain shower/ Light rain 
# 10 = rain with thunderstorm 
# 26 = snowfall 
# 94 = Freezing Fog



df_raw = pd.read_csv('london_merged.csv', delimiter = ',') 
df_raw.shape
df_raw.describe()
df_raw.head()
df_raw.dtypes
df_cl = df_raw
df_cl['timestamp'] = pd.to_datetime(df_cl['timestamp'])
df_cl['year'] = df_cl['timestamp'].dt.year
df_cl['day'] = df_cl['timestamp'].dt.day
df_cl.head() 

# %% For the second session :  Data issues 

describe= df_raw.describe()
df_raw.isnull().sum() 

# %% For the third session :  Data visualization 
from matplotlib.pyplot import figure, subplot, hist, xlabel,show
from matplotlib.pyplot import boxplot, xticks, ylabel, title
from matplotlib.pyplot import plot, yticks,legend 
from scipy import stats


data = pd.read_csv("london_merged.csv")
data['weather_code'] = data['weather_code'].rank(method='dense').astype(int)
data['timestamp'] = pd.to_datetime(data['timestamp'], format ="%Y-%m-%d %H:%M:%S")
data['hour'] = data['timestamp'].dt.hour
data['month'] = data['timestamp'].dt.month
data['year'] = data['timestamp'].dt.year
data = data.drop('timestamp', 1)
data = data[['cnt', 't1', 't2', 'hum', 'wind_speed', 'hour', 'month', 'year', 'season', 'weather_code', 'is_holiday', 'is_weekend']]
attributeNames = data.columns.values


#data = data.loc[data['year'] == 2017]
X = data.to_numpy()
y = data['season'].to_numpy()
data.head()

classNames = sorted(set(y))
N = len(y)
M_real = len(attributeNames)
M = 5
C = len(classNames)
nbins = 24


figure(figsize=(10,10))
u = np.floor(np.sqrt(M)); v = np.ceil(float(M)/u)
for i in range(M):
    subplot(u,v,i+1)
    T = X[:,i]
    hist(T, bins = nbins)
    xlabel(attributeNames[i])    
show()


figure(figsize=(10,10))
u = np.floor(np.sqrt(M)); v = np.ceil(float(M)/u)
for i in range(M):
    subplot(u,v,i+1)
    T = X[:,i]
    hist(T, bins = nbins, density = True)
    xlabel(attributeNames[i])
    mean = T.mean()
    std = T.std(ddof = 1)
    t = np.linspace(T.min(), T.max(), 1000)
    pdf = stats.norm.pdf(t,mean,std)
    plot(t,pdf,'.',color='red')
    #print(pdf)
    #ylim(0,N/2)
    
show()


figure(figsize=(10,10))
u = np.floor(np.sqrt(M)); v = np.ceil(float(M)/u)
for i in range(M):
    subplot(u,v,i+1)
    boxplot(X[:,i])
    xlabel(attributeNames[i])
    #ylim(0,N/2)
    
show()

figure(figsize=(24,10))
u = np.floor(np.sqrt(M)); v = np.ceil(float(M)/u)
for i in range(M):
    subplot(u,v,i+1)
    for c in range(C):
        class_mask = (y==c) # binary mask to extract elements of class c
        boxplot(X[class_mask,i], positions = [c + 1])
    title(attributeNames[i])
    xticks(range(1, C + 1), classNames)
    
show()


print("Correlations:")
for m1 in range(M):
    for m2 in range(m1 + 1, M):
        cor = np.cov(X[:,m1].T, X[:,m2].T)/(np.std(X[:,m1]) * np.std(X[:,m2]))
        print(attributeNames[m1] , "\t" ,attributeNames[m2], ":\t",  cor[0][1])
        

figure(figsize=(12,10))
for m1 in range(M):
    for m2 in range(M):
        subplot(M, M, m1*M + m2 + 1)
        for c in range(C):
            class_mask = (y==c)
            plot(np.array(X[class_mask,m2]), np.array(X[class_mask,m1]), '.')
            if m1==M-1:
                xlabel(attributeNames[m2])
            else:
                xticks([])
            if m2==0:
                ylabel(attributeNames[m1])
            else:
                yticks([])
            #ylim(0,X.max()*1.1)
            #xlim(0,X.max()*1.1)
legend(classNames)


from matplotlib.pyplot import (figure, imshow, xticks, xlabel, ylabel, title, 
                               colorbar, cm, show)
from scipy.stats import zscore


X_standarized = zscore(X, ddof=1)[:, :M]

figure(figsize=(12,6))
imshow(X_standarized, interpolation='none', aspect=(4./N), cmap=cm.gray);
xticks(range(M), attributeNames)
xlabel('Attributes')
ylabel('Data objects')
title('Fisher\'s Iris data matrix')
colorbar()

show()

# %% For the third session : PCA

# %% Dataset manipulations
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


# %% The amount of variation explained as a function of the number of PCA components included
# Subtract mean value from data
Y = X_new - np.ones((N,1))*np.array(X_new.mean(axis=0))
Y2 = Y*(1/np.std(Y,0))
# PCA by computing SVD of Y
titles = ['Zero-mean', 'Zero-mean and unit variance']
k=[Y,Y2]
plt.figure(figsize=(10,5))
for i in [0,1]:
    U,S,Vh = svd(k[i],full_matrices=False)
    
    # Compute variance explained by principal components
    rho = (S*S) / (S*S).sum() 
    threshold = 0.9
    
    # Plot variance explained
    plt.subplot(1, 2, 1+i)
    plt.plot(range(1,len(rho)+1),rho,'x-')
    plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
    plt.plot([1,len(rho)],[threshold, threshold],'k--')
    plt.title('Variance explained by principal components');
    plt.xlabel('Principal component');
    plt.ylabel('Variance explained value');
    plt.legend(['Individual','Cumulative','Threshold'])
    plt.title(titles[i])
    plt.grid()    
plt.show()


# %% plot the principal directions of the considered PCA components 
# Plot attribute coefficients in principal component space
i = 0
j = 1
V = Vh.T
attributeNames = X_new.columns.values

for att in range(V.shape[1]):
    plt.arrow(0,0, V[att,i], V[att,j])
    plt.text(V[att,i], V[att,j], attributeNames[att])
plt.xlim([-1,1])
plt.ylim([-1,1])
plt.xlabel('PC'+str(i+1))
plt.ylabel('PC'+str(j+1))
plt.grid()
 # Add a unit circle
plt.plot(np.cos(np.arange(0, 2*np.pi, 0.01)), 
         np.sin(np.arange(0, 2*np.pi, 0.01)));
plt.title('Attribute coefficients')
plt.axis('equal')
plt.show()


# %% Project the centered data onto principal component space
Z = np.dot(Y,V)
# Indices of the principal components to be plotted
# Plot PCA of the data
plt.title('london-bike-sharing-dataset: PCA')
plt.plot(Z[:48,i], Z[:48,j], 'o', alpha=.5)
plt.plot(Z[:24,i], Z[:24,j], 'o')
plt.xlabel('PC{0}'.format(i+1))
plt.ylabel('PC{0}'.format(j+1))
plt.legend(['date2','date1'])





