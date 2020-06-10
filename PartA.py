#!/usr/bin/env python
# coding: utf-8

# %% For the first session : A description of your data set
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

from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid)
import numpy as np
from scipy.io import loadmat
import sklearn.linear_model as lm
from sklearn import model_selection
from toolbox_02450 import rlr_validate

N, M = X.shape

# Add offset attribute
X_ANN = np.copy(X)
y_ANN = np.copy(y)
X = np.concatenate((np.ones((X.shape[0],1)),X),1)
attributeNames = [u'Offset']+attributeNames
M = M+1

## Crossvalidation
# Create crossvalidation partition for evaluation
K = 10
CV = model_selection.KFold(K, shuffle=True)
#CV = model_selection.KFold(K, shuffle=False)

#lambdas = range(1,100)
lambdas = np.power(10.,range(-5,9))

# Initialize variables
#T = len(lambdas)
Error_train = np.empty((K,1))
Error_test = np.empty((K,1))
Error_train_rlr = np.empty((K,1))
Error_test_rlr = np.empty((K,1))
Error_train_nofeatures = np.empty((K,1))
Error_test_nofeatures = np.empty((K,1))
w_rlr = np.empty((M,K))
mu = np.empty((K, M-1))
sigma = np.empty((K, M-1))
w_noreg = np.empty((M,K))

k=0
for train_index, test_index in CV.split(X,y):
    
    # extract training and test set for current CV fold
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    internal_cross_validation = 10    
    
    opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train, y_train, lambdas, internal_cross_validation)

    # Standardize outer fold based on training set, and save the mean and standard
    # deviations since they're part of the model (they would be needed for
    # making new predictions) - for brevity we won't always store these in the scripts
    print(mu[k, :].shape)
    mu[k, :] = np.mean(X_train[:, 1:], 0)
    sigma[k, :] = np.std(X_train[:, 1:], 0)
    
    X_train[:, 1:] = (X_train[:, 1:] - mu[k, :] ) / sigma[k, :] 
    X_test[:, 1:] = (X_test[:, 1:] - mu[k, :] ) / sigma[k, :] 
    
    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train
    
    # Compute mean squared error without using the input data at all
    Error_train_nofeatures[k] = np.square(y_train-y_train.mean()).sum(axis=0)/y_train.shape[0]
    Error_test_nofeatures[k] = np.square(y_test-y_test.mean()).sum(axis=0)/y_test.shape[0]

    # Estimate weights for the optimal value of lambda, on entire training set
    lambdaI = opt_lambda * np.eye(M)
    lambdaI[0,0] = 0 # Do no regularize the bias term
    w_rlr[:,k] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
    # Compute mean squared error with regularization with optimal lambda
    Error_train_rlr[k] = np.square(y_train-X_train @ w_rlr[:,k]).sum(axis=0)/y_train.shape[0]
    Error_test_rlr[k] = np.square(y_test-X_test @ w_rlr[:,k]).sum(axis=0)/y_test.shape[0]

    # Display the results for the last cross-validation fold
    if k == K-1:
        figure(k, figsize=(12,8))
        subplot(1,2,1)
        semilogx(lambdas,mean_w_vs_lambda.T[:,1:],'.-') # Don't plot the bias term
        xlabel('Regularization factor')
        ylabel('Mean Coefficient Values')
        grid()
        # You can choose to display the legend, but it's omitted for a cleaner 
        # plot, since there are many attributes
        #legend(attributeNames[1:], loc='best')
        
        subplot(1,2,2)
        title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
        loglog(lambdas,train_err_vs_lambda.T,'b.-',lambdas,test_err_vs_lambda.T,'r.-')
        xlabel('Regularization factor')
        ylabel('Squared error (crossvalidation)')
        legend(['Train error','Validation error'])
        grid()
    

    k+=1

show()