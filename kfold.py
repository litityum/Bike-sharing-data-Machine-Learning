#Multinomial regression project 2

from sklearn.model_selection import KFold # import KFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import sklearn.linear_model as lm
import pylab
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
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
#K = season.max()+1
#season_encoding = np.zeros((season.size, K))
#season_encoding[np.arange(season.size), season] = 1

#X_new = X_new.join(pd.DataFrame(season_encoding,
 #                  columns=['winter','spring','summer','autumn']))
X = X_new.to_numpy()
y = season
#print(y.shape)
attributeNames = list(X_new.columns)
classNames= ['winter','spring','summer','autumn'] 

#print(type(X))
#print(type(y))
#print(X.shape)
#print(y.shape)

# X = X - np.ones((X.shape[0],1)) * np.mean(X,0)
N, M = X.shape
C = len(classNames)

###########################################################################3
lambdas =np.logspace(-8, 4, 50)
error= []
train_error_rate = np.zeros(len(lambdas))
test_error_rate = np.zeros(len(lambdas))
coefficient_norm = np.zeros(len(lambdas))

no_splits= 4
kf = KFold(n_splits=no_splits)
kf.get_n_splits(X)
a= kf.split(X)
print(a) 
#print(kf)
for k in range(0, len(lambdas)):
    regularization_strength = lambdas[k]
    mdl = lm.LogisticRegression(solver='lbfgs', multi_class='multinomial', 
                                   tol=1e-8, random_state=1, 
                                   penalty='l2', C=1/regularization_strength)
    gen_error= 0 
    train_error_rate1 = np.zeros(no_splits)
    test_error_rate1 = np.zeros(no_splits)
    i=0   
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        mu = np.mean(X_train, 0)
        sigma = np.std(X_train, 0)
        X_train = (X_train - mu) / sigma
        X_test = (X_test - mu) / sigma
        
        mdl.fit(X_train, y_train)
        y_train_est = mdl.predict(X_train).T
        y_test_est = mdl.predict(X_test).T
        train_error_rate1[i] = train_error_rate1[i]+ np.sum(y_train_est != y_train) / len(y_train)
        test_error_rate1[i] = test_error_rate1[i]+ np.sum(y_test_est!=y_test) / len(y_test)
        gen_error = gen_error + np.sum(y_test_est!=y_test)
        i+=1
        
    train_error_rate[k]= np.mean(train_error_rate1)
    test_error_rate[k] = np.mean(test_error_rate1)
    gen_error = gen_error/17414
    w_est = mdl.coef_[0]
    coefficient_norm[k] = np.sqrt(np.sum(w_est**2))
    #print(reg)
    error.append(gen_error)
    
min_error = np.min(test_error_rate)
opt_lambda_idx = np.argmin(test_error_rate)
opt_lambda = lambdas[opt_lambda_idx]
    #print(gen_error)

##subplot(1,2,1)
#title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
#loglog(lambdas,error.T,'b.-',lambdas,error.T,'r.-')
#xlabel('Regularization factor')
#ylabel('Squared error (crossvalidation)')
#legend(['Train error','Validation error'])
#plt.show()
    
    
#fig = plt.figure()
#ax = fig.add_subplot(1, 1, 1)
#line = ax.plot(error, color='blue', lw=2)
#ax.set_xscale('log')
#pylab.show()
    
plt.figure(figsize=(8,8))
#plt.plot(np.log10(lambda_interval), train_error_rate*100)
#plt.plot(np.log10(lambda_interval), test_error_rate*100)
#plt.plot(np.log10(opt_lambda), min_error*100, 'o')
print(train_error_rate)
print(test_error_rate)
plt.semilogx(lambdas, train_error_rate*100)
plt.semilogx(lambdas, test_error_rate*100)
plt.semilogx(opt_lambda, min_error*100, 'o')
plt.text(1e-8, 45, "Minimum test error: " + str(np.round(min_error*100,2)) + ' % at 1e' + str(np.round(np.log10(opt_lambda),2)))
plt.xlabel('Regularization strength, $\log_{10}(\lambda)$')
plt.ylabel('Error rate (%)')
plt.title('Classification error')
plt.legend(['Training error','Test error','Test minimum'],loc='upper right')
plt.ylim([30, 70])
plt.grid()
plt.show()    

plt.figure(figsize=(8,8))
plt.semilogx(lambdas, coefficient_norm,'k')
plt.ylabel('L2 Norm')
plt.xlabel('Regularization strength, $\log_{10}(\lambda)$')
plt.title('Parameter vector L2 norm')
plt.grid()
plt.show()    