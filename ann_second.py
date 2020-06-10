# exercise 8.3.1 Fit neural network classifiers using softmax output weighting
from matplotlib.pyplot import figure, show, title
from scipy.io import loadmat
from toolbox_02450 import dbplotf, train_neural_net, visualize_decision_boundary
import numpy as np
import torch
# Load Matlab data file and extract variables of interest
#mat_data = loadmat('../Data/synth1.mat')
#X = mat_data['X']
#X = X - np.ones((X.shape[0],1)) * np.mean(X,0)
#X_train = mat_data['X_train']
#X_test = mat_data['X_test']
#y = mat_data['y'].squeeze()
#y_train = mat_data['y_train'].squeeze()
#y_test = mat_data['y_test'].squeeze()
#
#print(X.shape)
#print(X_train.shape)
#print(X_test.shape)
#
#print(y_train.shape)
#print(y_test.shape)
#print(y.shape)
#print(type(y))
#print(type(y_train))
#print(type(y_test))
#
#attributeNames = [name[0] for name in mat_data['attributeNames'].squeeze()]
#classNames = [name[0][0] for name in mat_data['classNames']]
#
#N, M = X.shape
#C = len(classNames)

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

#X_new = X_new.join(pd.DataFrame(season_encoding,
                   #columns=['winter','spring','summer','autumn']))




X = X_new.to_numpy()
y = season
N,M=X.shape
attributeNames = list(X_new.columns)
classNames= ['winter','spring','summer','autumn'] 
#%% Model fitting and prediction

kf = KFold(n_splits=5)
error= np.zeros(5)
# Define the model structure
i=0
for train_index, test_index in kf.split(X):

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    print(y_test_est.shape)
        #%% Model fitting and predictio
    n_hidden_units = 5 # number of hidden units in the signle hidden layer
    model = lambda: torch.nn.Sequential(
                                torch.nn.Linear(13, n_hidden_units), #M features to H hiden units
                                torch.nn.ReLU(), # 1st transfer function
                                # Output layer:
                                # H hidden units to C classes
                                # the nodes and their activation before the transfer 
                                # function is often referred to as logits/logit output
                                torch.nn.Linear(n_hidden_units, C), # C logits
                                # To obtain normalised "probabilities" of each class
                                # we use the softmax-funtion along the "class" dimension
                                # (i.e. not the dimension describing observations)
                                torch.nn.Softmax(dim=1) # final tranfer function, normalisation of logit output
                                )
    # Since we're training a multiclass problem, we cannot use binary cross entropy,
    # but instead use the general cross entropy loss:
    loss_fn = torch.nn.CrossEntropyLoss()
    # Train the network:
    net, _, _ = train_neural_net(model, loss_fn,
                                 X=torch.tensor(X_train, dtype=torch.float),
                                 y=torch.tensor(y_train, dtype=torch.long),
                                 n_replicates=3,
                                 max_iter=10000)
    # Determine probability of each class using trained network
    softmax_logits = net(torch.tensor(X_test, dtype=torch.float))
    # Get the estimated class as the class with highest probability (argmax on softmax_logits)
    y_test_est = (torch.max(softmax_logits, dim=1)[1]).data.numpy() 
    # Determine errors
    e = (y_test_est != y_test)
    print('Number of miss-classifications for ANN:\n\t {0} out of {1}'.format(sum(e),len(e)))
    error[i] = np.sum(e)/len(y_test)
    predict = lambda x:  (torch.max(net(torch.tensor(x, dtype=torch.float)), dim=1)[1]).data.numpy() 
    figure(i,figsize=(9,9))
    visualize_decision_boundary(predict, [X_train, X_test], [y_train, y_test], attributeNames, classNames)
    title('ANN decision boundaries')

show()
print(error)
print('Ran Exercise 8.3.1')