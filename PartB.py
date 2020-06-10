#!/usr/bin/env python
# coding: utf-8

# %% For the first session : A description of your data set
import pandas as pd
import numpy as np


# %% For the third session :  Data visualization 
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, subplot, hist, xlabel,show
from matplotlib.pyplot import boxplot, xticks, ylabel, title
from matplotlib.pyplot import plot, yticks,legend 
from matplotlib.pylab import semilogx, loglog, ylabel,subplot, grid

from scipy import stats
from scipy.linalg import svd
from scipy.io import loadmat

import sklearn.linear_model as lm
from sklearn import model_selection

import torch

from toolbox_02450 import rlr_validate
from toolbox_02450 import train_neural_net, draw_neural_net


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




N, M = X.shape

X_ANN = stats.zscore(X)
y_ANN = np.copy(y).reshape(y.shape[0], 1)
X = np.concatenate((np.ones((X.shape[0],1)),X),1)
attributeNames = [u'Offset']+attributeNames
M = M+1




K1, K2 = 5, 5
CV1 = model_selection.KFold(K1, shuffle=True)

hidden_array = range(1,6)

N, M_ANN = X_ANN.shape

n_replicates = 1        # number of networks trained in each k-fold
max_iter = 5000


lambdas = np.power(10.,range(-5,9))

# Initialize variables
#T = len(lambdas)
Final_ANN_Error = np.empty((K1,1))
Hidden_ANN = np.empty((K1,1))
Opt_lambda_arr = np.empty((K1,1))
Error_test_rlr = np.empty((K1,1))
Error_test_nofeatures = np.empty((K1,1))
w_rlr = np.empty((M,K1))
mu = np.empty((K1, M-1))
sigma = np.empty((K1, M-1))
w_noreg = np.empty((M,K1))

r1 = []
r2 = []
r3 = []

k = 0
for train_index, test_index in CV1.split(X,y):
    print("k = ", k)
    
    # extract training and test set for current CV fold
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    
    X_ANN_train = X_ANN[train_index]
    y_ANN_train = y_ANN[train_index]
    X_ANN_test = X_ANN[test_index]
    y_ANN_test = y_ANN[test_index]
    
    opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train, y_train, lambdas, K2)
    
    Opt_lambda_arr[k] = opt_lambda
    
    # Standardize outer fold based on training set, and save the mean and standard
    # deviations since they're part of the model (they would be needed for
    # making new predictions) - for brevity we won't always store these in the scripts
    mu[k, :] = np.mean(X_train[:, 1:], 0)
    sigma[k, :] = np.std(X_train[:, 1:], 0)
    
    X_train[:, 1:] = (X_train[:, 1:] - mu[k, :] ) / sigma[k, :] 
    X_test[:, 1:] = (X_test[:, 1:] - mu[k, :] ) / sigma[k, :] 
    
    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train
    
    # Compute mean squared error without using the input data at all
    Error_test_nofeatures[k] = np.square(y_test-y_test.mean()).sum(axis=0)/y_test.shape[0]

    # Estimate weights for the optimal value of lambda, on entire training set
    lambdaI = opt_lambda * np.eye(M)
    lambdaI[0,0] = 0 # Do no regularize the bias term
    w_rlr[:,k] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
    # Compute mean squared error with regularization with optimal lambda
    Error_test_rlr[k] = np.square(y_test-X_test @ w_rlr[:,k]).sum(axis=0)/y_test.shape[0]
    
    
    
    CV2 = model_selection.KFold(K2, shuffle=True)
    
    
   
    
    Error_ANN = np.zeros(len(hidden_array))
    for ANN_train_index, ANN_test_index in CV2.split(X_ANN_train,y_ANN_train):
        X_inner_train = torch.Tensor(X_ANN_train[ANN_train_index,:])
        y_inner_train = torch.Tensor(y_ANN_train[ANN_train_index])
        X_inner_test = torch.Tensor(X_ANN_train[ANN_test_index,:])
        y_inner_test = torch.Tensor(y_ANN_train[ANN_test_index])
        
        
        j = 0
        
        for n_hidden_units in hidden_array:
            print("hidden unit = ", n_hidden_units)
            model = lambda: torch.nn.Sequential(
                    torch.nn.Linear(M_ANN, n_hidden_units), #M features to n_hidden_units
                    torch.nn.ReLU(),   # 1st transfer function,
                    torch.nn.Linear(n_hidden_units, 1), # n_hidden_units to 1 output neuron
                    )
            
            loss_fn = torch.nn.MSELoss()
            net, final_loss, learning_curve = train_neural_net(model,loss_fn,X=X_inner_train,y=y_inner_train,n_replicates=n_replicates, max_iter=max_iter)
            test_est = net(X_inner_test)
    
            # Determine errors and errors
            se = (test_est.float()-y_inner_test.float())**2 # squared error
            mse = (sum(se).type(torch.float)/len(y_inner_test)).data.numpy() #mean
            Error_ANN[j] += mse[0]
            j += 1
    
    Hidden_ANN[k] = hidden_array[int(np.where(Error_ANN == np.amin(Error_ANN))[0])]
    
    model = lambda: torch.nn.Sequential(
                    torch.nn.Linear(M_ANN, int(Hidden_ANN[k])), #M features to n_hidden_units
                    torch.nn.ReLU(),   # 1st transfer function,
                    torch.nn.Linear(int(Hidden_ANN[k]), 1), # n_hidden_units to 1 output neuron
                    )
    net, final_loss, learning_curve = train_neural_net(model,
                                                       loss_fn,
                                                       X=torch.Tensor(X_ANN_test),
                                                       y=torch.Tensor(y_ANN_test),
                                                       n_replicates=n_replicates,
                                                       max_iter=max_iter)
    y_test_est = net(torch.Tensor(X_ANN_test))
    
    # Determine errors and errors
    se = (y_test_est.float()-torch.Tensor(y_ANN_test).float())**2 # squared error
    mse = (sum(se).type(torch.float)/len(y_ANN_test)).data.numpy() #mean
    Final_ANN_Error[k] = mse[0] # store error rate for current CV fold
    
    

    k+=1
    
for k in range(K1):
    print(k, "\t", int(Hidden_ANN[k]), "\t", Final_ANN_Error[k], "\t", Opt_lambda_arr[k], "\t", Error_test_rlr[k], Error_test_nofeatures[k])
    r1.append(Error_test_rlr[k] - Error_test_nofeatures[k])
    r2.append(Error_test_nofeatures[k] - Final_ANN_Error[k])
    r3.append(Final_ANN_Error[k] - Error_test_rlr[k])
    
alpha = 0.05
rho = 1/K1
p1_setupII, CI1_setupII = correlated_ttest(r1, rho, alpha=alpha)
p2_setupII, CI2_setupII = correlated_ttest(r2, rho, alpha=alpha)
p3_setupII, CI3_setupII = correlated_ttest(r3, rho, alpha=alpha)

print(p1_setupII, " ", p2_setupII, " ", p3_setupII)
print(CI1_setupII)
print(CI2_setupII)
print(CI3_setupII)



