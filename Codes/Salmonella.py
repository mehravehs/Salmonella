# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 19:40:42 2017

@author: Mehraveh
"""


%reset

from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale


import scipy
import imp
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier  #GBM algorithm
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.ensemble import GradientBoostingClassifier  #GBM algorithm
import matplotlib
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import OrthogonalMatchingPursuitCV
from sklearn.datasets import make_sparse_coded_signal
from sklearn.metrics import classification_report
import plotly.plotly as py
from plotly.tools import FigureFactory as FF
from sklearn.manifold import TSNE
from sklearn import manifold
from sklearn.metrics.pairwise import euclidean_distances
import scipy as sc
import pylab
import scipy
import imp
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier  #GBM algorithm
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.ensemble import GradientBoostingClassifier  #GBM algorithm
import matplotlib
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import OrthogonalMatchingPursuitCV
from sklearn.datasets import make_sparse_coded_signal
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor  #GBM algorithm
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.cross_validation import LeaveOneOut, cross_val_predict
import multiprocessing
from sklearn.metrics import confusion_matrix
from itertools import combinations
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist, squareform


data = pd.read_csv('/Users/Mehraveh/Downloads/Species102017.csv', sep=',', delimiter=None, header=None)
data = data.fillna(0)



### Analaysis with all 9 classes
X=data.drop(data.index[data[0]==0]) # The first three rows in data are removed as they are meant to be unknown for the model
X=X.drop(0, 1) # The first column in data is the class names, which is removed from X. This is y.
X=np.asmatrix(X)

y=data[0] # first column of data
y = y.drop(y.index[data[0]==0])
y=y.ravel()


#X_bin = 1*(X>0)
n_samples, n_features = X.shape
n_digits=2

### Leave-one-samepl-out cross-validation model
y_pred = np.zeros(n_samples)
y_pred_bin = np.zeros(n_samples)
class_probs = np.zeros([n_samples,np.unique(y).size]) # the probability of assigning each left out sample to each of the classes
loo = LeaveOneOut(n_samples)
for train_index, test_index in loo:
    print(test_index)
    clf = GradientBoostingClassifier(n_estimators=100,learning_rate=0.05) 
    clf.fit(X[train_index,:],y[train_index])
    y_pred[test_index] = clf.predict(X[test_index,:])    
    class_probs[test_index,:] = clf.predict_proba(X[test_index,:])

#    clf = GradientBoostingClassifier(n_estimators=100,learning_rate=0.05) 
#    clf.fit(X_bin[train_index,:],y[train_index])
#    y_pred_bin[test_index] = clf.predict(X_bin[test_index,:])    

    
my_score = np.mean(y_pred==y)
#my_score_bin = np.mean(y_pred_bin==y)
print(my_score)
#print(my_score_bin)
prob_loo = class_probs
plt.imshow(prob_loo, cmap=plt.cm.coolwarm, interpolation='none', extent=[0,150,0,np.size(prob_loo,0)])
plt.grid(True)
plt.xticks([])
plt.yticks([])
ax = plt.gca();
ax.grid(color='w', linestyle='-', linewidth=0)
plt.colorbar()
plt.savefig('/Users/Mehraveh/Desktop/class_probs_leave_one_out.png', dpi=120)

### What are the three unknown classes?
X_test=data.ix[data[0]==0,:]
X_test=X_test.drop(0, 1)
X_test=np.asmatrix(X_test)
y_test=data[0]
y_test = y_test.ix[data[0]==0]
y=y.ravel()
X_train = X
y_train = y
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

class_probs_unknowns = clf.predict_proba(X_test)
prob_loo_unknowns = class_probs_unknowns
scipy.io.savemat('/Users/Mehraveh/Documents/MATLAB/Reza/probability_all_9.mat', mdict={'prob_loo': prob_loo, 'correctY': y , 'prob_loo_unknowns': prob_loo_unknowns})


### Analaysis with 8 classes after removing contaminated milk
X=data.drop(data.index[np.where(np.logical_or(data[0]==8,data[0]==0))])
X=X.drop(0, 1)
X=np.asmatrix(X)

y=data[0]
y = y.drop(y.index[np.where(np.logical_or(data[0]==8,data[0]==0))])
y=y.ravel()

#X_bin = 1*(X>0)

n_samples, n_features = X.shape
n_digits=2

### Leave-one-samepl-out cross-validation model
y_pred = np.zeros(n_samples)
y_pred_bin = np.zeros(n_samples)
class_probs = np.zeros([n_samples,np.unique(y).size])
loo = LeaveOneOut(n_samples)
for train_index, test_index in loo:
    print(test_index)
    clf = GradientBoostingClassifier(n_estimators=100,learning_rate=0.05) 
    clf.fit(X[train_index,:],y[train_index])
    y_pred[test_index] = clf.predict(X[test_index,:])    
    class_probs[test_index,:] = clf.predict_proba(X[test_index,:])

#    clf = GradientBoostingClassifier(n_estimators=100,learning_rate=0.05) 
#    clf.fit(X_bin[train_index,:],y[train_index])
#    y_pred_bin[test_index] = clf.predict(X_bin[test_index,:])    

my_score = np.mean(y_pred==y)
#my_score_bin = np.mean(y_pred_bin==y)
print(my_score)
#print(my_score_bin)
prob_loo = class_probs
plt.imshow(prob_loo, cmap=plt.cm.coolwarm, interpolation='none', extent=[0,150,0,np.size(prob_loo,0)])
plt.grid(True)
plt.xticks([])
plt.yticks([])
ax = plt.gca();
ax.grid(color='w', linestyle='-', linewidth=0)
plt.colorbar()
plt.savefig('/Users/Mehraveh/Desktop/class_probs_leave_one_out.png', dpi=120)


### What is contaminated milk assigned to?
X_test=data.ix[data[0]==8,:]
X_test=X_test.drop(0, 1)
X_test=np.asmatrix(X_test)
y_test=data[0]
y_test = y_test.ix[data[0]==8]
y=y.ravel()
X_train = X
y_train = y
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

class_probs_contaminated = clf.predict_proba(X_test)
prob_loo_contaminated = class_probs_contaminated


### What are the three unknown classes?

X_test=data.ix[data[0]==0,:]
X_test=X_test.drop(0, 1)
X_test=np.asmatrix(X_test)
y_test=data[0]
y_test = y_test.ix[data[0]==0]
y_test=y_test.ravel()
X_train = X
y_train = y
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
class_probs_unknowns = clf.predict_proba(X_test)
prob_loo_unknowns = class_probs_unknowns

scipy.io.savemat('/Users/Mehraveh/Documents/MATLAB/Reza/probability_8.mat', mdict={'prob_loo': prob_loo, 'correctY': y , 'prob_loo_contaminated': prob_loo_contaminated, 'prob_loo_unknowns': prob_loo_unknowns})

