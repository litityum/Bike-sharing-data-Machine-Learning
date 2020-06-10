from sklearn.model_selection import KFold # import KFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics
from sklearn import datasets
from sklearn.model_selection import train_test_split
import sklearn.linear_model as lm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection

from toolbox_02450 import *
# requires data from exercise 1.5.1
#from ex5_1_5 import *

import numpy as np
import pylab
import torch
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, subplot, hist, xlabel,show
from matplotlib.pyplot import boxplot, xticks, ylabel, title
from matplotlib.pyplot import plot, yticks,legend 
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, show, title

from scipy import stats
from scipy.linalg import svd
from scipy.io import loadmat

from toolbox_02450 import dbplotf, train_neural_net, visualize_decision_boundary
from toolbox_02450 import rocplot, confmatplot


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

X = X_new.to_numpy()
y = season
attributeNames = list(X_new.columns)
classNames= ['winter','spring','summer','autumn'] 

N, M = X.shape
C = len(classNames)

K1, K2 = 5, 5
CV1 = KFold(K1, shuffle=True)
hidden_array = range(1,6)
#lambdas = np.power(10.,range(-5,9))

Opt_lambda_arr = np.zeros((K1,1))
Error_test_nofeatures = np.zeros((K1,1))
Error_test_rlr = np.zeros((K1,1))
Final_ANN_Error = np.zeros((K1,1))
Hidden_ANN = np.zeros((K1,1))

r1 = []
r2 = []
r3 = []


k=0
for train_index, test_index in CV1.split(X,y):
    #Outer cross-validation loop. First make the outer split into K1 folds
    print("k = ", k)
    
    X_train_twofold = X[train_index]
    y_train_twofold = y[train_index]
    X_test_twofold = X[test_index]
    y_test_twofold = y[test_index]
    
    mu = np.mean(X_train_twofold, 0)
    sigma = np.std(X_train_twofold, 0)
    X_train_twofold = (X_train_twofold - mu) / sigma
    X_test_twofold = (X_test_twofold - mu) / sigma
    
    #first method
    output = np.bincount(y_train_twofold).argmax() 
    output_array = output * np.ones(len(y_test_twofold))
    Error_test_nofeatures[k] = np.sum(output_array !=y_test_twofold) / len(y_test_twofold)
#    
#    #second method 
#    #finding the optimal lambda for X_train and X_test using and putting it as optical lambda [k[]]
#    #basically use X_train as the X for the whole code 
    lambdas =np.logspace(-8, 4, 50)
    train_error_rate = np.zeros(len(lambdas))
    test_error_rate = np.zeros(len(lambdas))
    coefficient_norm = np.zeros(len(lambdas))
    
    no_splits= 5
    kf = KFold(n_splits=no_splits)
    #kf.get_n_splits(X_train_twofold)
    #a= kf.split(X_train_twofold)
    print(a) 
    #print(kf)
    for k1 in range(0, len(lambdas)):
        regularization_strength = lambdas[k1]
        mdl = lm.LogisticRegression(solver='lbfgs', multi_class='multinomial', 
                                       tol=1e-8, random_state=1, 
                                       penalty='l2', C=1/regularization_strength)
        train_error_rate1 = np.zeros(no_splits)
        test_error_rate1 = np.zeros(no_splits)
        i=0   
        for train_index1, test_index1 in kf.split(X_train_twofold,y_train_twofold):
            X_train_inside, X_test_inside = X_train_twofold[train_index1], X_train_twofold[test_index1]
            y_train_inside, y_test_inside = y_train_twofold[train_index1], y_train_twofold[test_index1]
            
#            mu = np.mean(X_train, 0)
#            sigma = np.std(X_train, 0)
#            X_train = (X_train - mu) / sigma
#            X_test = (X_test - mu) / sigma
#            
            mdl.fit(X_train_inside, y_train_inside)
            y_train_est = mdl.predict(X_train_inside).T
            y_test_est = mdl.predict(X_test_inside).T
            train_error_rate1[i] = train_error_rate1[i]+ np.sum(y_train_est != y_train_inside) / len(y_train_inside)
            test_error_rate1[i] = test_error_rate1[i]+ np.sum(y_test_est!=y_test_inside) / len(y_test_inside)
            i+=1
            
        train_error_rate[k1]= np.mean(train_error_rate1)
        test_error_rate[k1] = np.mean(test_error_rate1)
        w_est = mdl.coef_[0]
        coefficient_norm[k1] = np.sqrt(np.sum(w_est**2))
        #print(reg)
        #error.append(gen_error)
        
    min_error = np.min(test_error_rate)
    opt_lambda_idx = np.argmin(test_error_rate)
    opt_lambda = lambdas[opt_lambda_idx]
        
        
    Opt_lambda_arr[k] =  opt_lambda
    Error_test_rlr[k] = min_error
   ##################################################################
    #third method
    #use Xtrain to get the n suitable 
    #CV2 = model_selection.KFold(K2, shuffle=True)
    no_splits1 = 5
    kf1 = KFold(n_splits=no_splits1)
    #kf1.get_n_splits(X_train_twofold) 
    #print(kf1)
    
    hidden =  range(1,4,16,64)
    train_error_rate_ANN = np.zeros(len(hidden))
    
    for h in range(0,len(hidden)):
        
        train_error_rate1_ANN = np.zeros(no_splits1)
        i=0
        for train_index2, test_index2 in kf1.split(X_train_twofold,y_train_twofold):
            X_train_inside_ANN, X_test_inside_ANN = X_train_twofold[train_index2], X_train_twofold[test_index2]
            y_train_inside_ANN, y_test_inside_ANN = y_train_twofold[train_index2], y_train_twofold[test_index2]
            
                       
#            mu = np.mean(X_train, 0)
#            sigma = np.std(X_train, 0)
#            X_train = (X_train - mu) / sigma
#            X_test = (X_test - mu) / sigma
#            #%% Model fitting and prediction
            
            # Define the model structure
            n_hidden_units = hidden[h] # number of hidden units in the signle hidden layer
            model = lambda: torch.nn.Sequential(
                                        torch.nn.Linear(M, n_hidden_units), #M features to H hiden units
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
                                         X=torch.tensor(X_train_inside_ANN, dtype=torch.float),
                                         y=torch.tensor(y_train_inside_ANN, dtype=torch.long),
                                         n_replicates=3,max_iter=5000)
    
            # Determine probability of each class using trained network
            softmax_logits = net(torch.tensor(X_test_inside_ANN, dtype=torch.float))
            # Get the estimated class as the class with highest probability (argmax on softmax_logits)
            y_test_est_ANN = (torch.max(softmax_logits, dim=1)[1]).data.numpy() 
            print(y_test_est_ANN.shape)
            print(y_test_inside_ANN.shape)
            # Determine errors
            e = (y_test_est_ANN != y_test_inside_ANN)
            train_error_rate1_ANN[i] = train_error_rate1_ANN[i]+ np.sum(y_test_est_ANN !=y_test_inside_ANN)/ len(y_test_inside_ANN)
           # print('Number of miss-classifications for ANN:\n\t {0} out of {1}'.format(sum(e),len(e)))
            
            predict = lambda x:  (torch.max(net(torch.tensor(x, dtype=torch.float)), dim=1)[1]).data.numpy() 
            #figure(1,figsize=(9,9))
            #visualize_decision_boundary(predict, [X_train, X_test], [y_train, y_test], attributeNames, classNames)
            #title('ANN decision boundaries')
            #gen_error = gen_error + np.sum(y_test_est!=y_test)
            i +=1
        
        train_error_rate_ANN[h] = np.mean(train_error_rate1_ANN)
        #gen_error = gen_error/17414
        #error.append(gen_error)
        
    #print(train_error_rate)
    min= np.argmin(train_error_rate_ANN)
    #print(hidden[min])
    Final_ANN_Error[k] = np.min(train_error_rate_ANN)
    Hidden_ANN [k] = min
    
    k+=1
  ##################################
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


print(Hidden_ANN)
print(Opt_lambda_arr)

print("Baseline")
print(Error_test_rlr)
print("Error ann")
print(Final_ANN_Error) 
print("Error regress")
print(Error_test_nofeatures)   
    
    