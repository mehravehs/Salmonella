# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 19:51:03 2018

@author: Mehraveh
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 19:40:42 2017

@author: Mehraveh
"""


#%reset

from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

from sklearn.metrics import precision_recall_fscore_support
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
from sklearn.model_selection import KFold
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
import seaborn as sns

#data = data.iloc[2:]
#data = data.iloc[:,1:]




label = pd.read_csv('/Users/Mehraveh/Documents/Reza/Salmonella/Species08302018_labels_PTMs.csv', sep=',', delimiter=None,header=None)
label = label.drop(0, 1)
label = label.iloc[0]


data = pd.read_csv('/Users/Mehraveh/Documents/Reza/Salmonella/Species08302018_clean_PTMs.csv', sep=',', delimiter=None,header=None)
data = data.fillna(0)



### Figure 1
#Figure 1: S. cerevesiae [fungai], S typhimurium [bacteria], H. Salinarium [archaea], HeLa [animalia], arabidupsis [plantae] (edited)

classnumbers = np.array([1,4,5,7,10])
last_class=np.asscalar(data.tail(1)[0])
data=data.iloc[np.where(np.in1d(data.iloc[:,0], classnumbers))] # The UNcontaminated milks are removed for this study


X=data
X=X.iloc[:,1:] # The first column in data is the class names, which is removed from X. This is y.
X=np.asmatrix(X).astype(float)
y=data.iloc[:,0] # first column of data
y=y.astype(int).ravel()

#for i in range(174):
#    for j in range(65):
#        if (type(X[i,j]) != float):
#            print(i,j)
            

#X_bin = 1*(X>0)
n_samples, n_features = X.shape
n_digits=2



### Leave-one-samepl-out cross-validation model
y_pred = np.zeros(n_samples)
#y_pred_bin = np.zeros(n_samples)
imp_features = np.zeros(n_features)
#class_probs = np.zeros([n_samples,np.unique(y).size]) # the probability of assigning each left out sample to each of the classes

loo = LeaveOneOut(n_samples)
#kf = KFold(n_splits=10)
#loo = kf.split(X)
for train_index, test_index in loo:
    print(test_index)
    clf = GradientBoostingClassifier(n_estimators=100,learning_rate=0.05) 
    clf.fit(X[train_index,:],y[train_index])
    imp_features = imp_features + clf.feature_importances_
    y_pred[test_index] = clf.predict(X[test_index,:])    
#    class_probs[test_index,:] = clf.predict_proba(X[test_index,:])


#    clf = GradientBoostingClassifier(n_estimators=100,learning_rate=0.05) 
#    clf.fit(X_bin[train_index,:],y[train_index])
#    y_pred_bin[test_index] = clf.predict(X_bin[test_index,:])    

    
my_score = np.mean(y_pred==y)
#my_score_bin = np.mean(y_pred_bin==y)
print(my_score)
precision, recall, fscore, support = precision_recall_fscore_support(y, y_pred)


A=[]
for i in range(len(label)):
    A.append(label[i+1])
sio.savemat('/Users/Mehraveh/Documents/Reza/Salmonella/Acc.mat',{'precision':precision, 'recall':recall})
sio.savemat('/Users/Mehraveh/Documents/Reza/Salmonella/label.mat',{'label':A})
sio.savemat('/Users/Mehraveh/Documents/Reza/Salmonella/imp_feature.mat',{'imp':imp_features})
