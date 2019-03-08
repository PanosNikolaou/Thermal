#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 13:43:31 2018

@author: takis
"""

from __future__ import division
import pandas as pd
from numpy import genfromtxt
import matplotlib.pyplot as plt

import glob

THRlist = []
BLUElist = []
GREENlist = []
REDlist = []
SEQGlist = []
SEQRlist = []
SEQGElist = []
NIRlist = []
GNDVIlist = []
NVDIlist = []
RENVDIlist =[]
NDSMlist = []
SLOPElist = []
TPIlist = []
ROUGHNESSlist = []

for file_ in glob.glob('split/csv/1-*.csv'):
    sdata = genfromtxt(file_, delimiter=',')
    THRlist.extend(sdata.ravel())
    
print("1_cv loaded")
    
for file_ in glob.glob('split/csv/2-*.csv'):
    sdata = genfromtxt(file_, delimiter=',')
    BLUElist.extend(sdata.ravel())

print("2_cv loaded")
    
for file_ in glob.glob('split/csv/3-*.csv'):
    sdata = genfromtxt(file_, delimiter=',')
    GREENlist.extend(sdata.ravel())    

print("3_cv loaded")
    
for file_ in glob.glob('split/csv/4-*.csv'):
    sdata = genfromtxt(file_, delimiter=',')
    REDlist.extend(sdata.ravel())    

print("4_cv loaded")

for file_ in glob.glob('split/csv/5-*.csv'):
    sdata = genfromtxt(file_, delimiter=',')
    SEQGlist.extend(sdata.ravel())
    
print("5_cv loaded")
    
for file_ in glob.glob('split/csv/6-*.csv'):
    sdata = genfromtxt(file_, delimiter=',')
    SEQRlist.extend(sdata.ravel())

print("6_cv loaded")
    
for file_ in glob.glob('split/csv/7-*.csv'):
    sdata = genfromtxt(file_, delimiter=',')
    SEQGElist.extend(sdata.ravel())    

print("7_cv loaded")
    
for file_ in glob.glob('split/csv/8-*.csv'):
    sdata = genfromtxt(file_, delimiter=',')
    NIRlist.extend(sdata.ravel())   

print("8_cv loaded")

for file_ in glob.glob('split/csv/9-*.csv'):
    sdata = genfromtxt(file_, delimiter=',')
    GNDVIlist.extend(sdata.ravel())   

print("9_cv loaded")
    
for file_ in glob.glob('split/csv/10-*.csv'):
    sdata = genfromtxt(file_, delimiter=',')
    NVDIlist.extend(sdata.ravel())   

print("10_cv loaded")

for file_ in glob.glob('split/csv/11-*.csv'):
    sdata = genfromtxt(file_, delimiter=',')
    RENVDIlist.extend(sdata.ravel())   

print("11_cv loaded")

for file_ in glob.glob('split/csv/12-*.csv'):
    sdata = genfromtxt(file_, delimiter=',')
    NDSMlist.extend(sdata.ravel())       

print("12_cv loaded")

for file_ in glob.glob('split/csv/13-*.csv'):
    sdata = genfromtxt(file_, delimiter=',')
    SLOPElist.extend(sdata.ravel())   

print("13_cv loaded")
    
for file_ in glob.glob('split/csv/14-*.csv'):
    sdata = genfromtxt(file_, delimiter=',')
    TPIlist.extend(sdata.ravel())   

print("14_cv loaded")
    
for file_ in glob.glob('split/csv/15-*.csv'):
    sdata = genfromtxt(file_, delimiter=',')
    ROUGHNESSlist.extend(sdata.ravel())   

print("15_cv loaded")

print("READ CSV FILES COMPLETED")
    
bands_list = pd.DataFrame(
    {'THERMAL': THRlist,
     'BLUE': BLUElist,
     'GREEN': GREENlist,
     'RED': REDlist,
     'SEQGREEN': SEQGlist,
     'SEQRED': SEQRlist,
     'SEQREDEDGE': SEQGElist,
     'NIR': NIRlist,
     'GNDVI' : GNDVIlist,
     'NVDI' :NVDIlist,
     'RENVDI': RENVDIlist,
     'NDSM': NDSMlist,
     'SLOPE': SLOPElist,
     'TPI': TPIlist,
     'ROUGHNESS' :ROUGHNESSlist
     
     })

bands_list = bands_list.drop(bands_list[(
        bands_list.THERMAL == -999) 
        | (bands_list.RED == -999) 
        | (bands_list.BLUE == -999) 
        | (bands_list.GREEN == -999)
        | (bands_list.SEQGREEN == -999) 
        | (bands_list.SEQRED == -999) 
        | (bands_list.SEQREDEDGE == -999)
        | (bands_list.NIR == -999) 
        | (bands_list.GNDVI == -999) 
        | (bands_list.RENVDI == -999) 
        | (bands_list.NDSM == -999)
        | (bands_list.SLOPE == -999) 
        | (bands_list.TPI == -999) 
        | (bands_list.ROUGHNESS == -999)
        ].index)

print("CREATING BANDS COMPLETED")
    
###################################################
### DATA NORMALIZATION
###################################################

from pandas import Series
from sklearn.preprocessing import MinMaxScaler

THRseries = Series(bands_list.THERMAL)
THRvalues = THRseries.values
THRvalues = THRvalues.reshape((len(THRvalues), 1))
THRscaler = MinMaxScaler(feature_range=(0, 1000))
THRscaler = THRscaler.fit(THRvalues)
normalized = THRscaler.transform(THRvalues)
bands_list.THERMAL = normalized

#################################################
#NIRseries = Series(bands_list.THERMAL)
#NIRvalues = NIRseries.values
#NIRvalues = NIRvalues.reshape((len(NIRvalues), 1))
#NIRscaler = MinMaxScaler(feature_range=(0, 100))
#NIRscaler = NIRscaler.fit(NIRvalues)
#NIRnormalized = NIRscaler.transform(NIRvalues)
#bands_list.NIR = NIRnormalized
#################################################


import h2o
from h2o.estimators import H2ODeepLearningEstimator

h2o.init()

h2o.cluster().show_status()

data = h2o.H2OFrame(bands_list)
#print(data.as_data_frame())


splits = data.split_frame(ratios=[0.7, 0.15], seed=1)  

train = splits[0]
valid = splits[1]
test  = splits[2]

nfolds = 10
fold_assignment = 'Random'

#For Regression problems:
#
#    A Gaussian distribution is the function for continuous targets.
#    A Poisson distribution is used for estimating counts.
#    A Gamma distribution is used for estimating total values (such as claim payouts, rainfall, etc.).
#    A Tweedie distribution is used for estimating densities.
#    A Laplacian loss function (absolute L1-loss function) can predict the median percentile.
#    A Quantile regression loss function can predict a specified percentile.
#    A Huber loss function, a combination of squared error and absolute error, is more robust to outliers than L2 squared-loss function.

# next Gamma

#activation function (Tanh, Tanh with dropout, Rectifier, Rectifier with dropout, Maxout, Maxout with dropout).

#hidden_dropout_ratios: (Applicable only if the activation type is TanhWithDropout, RectifierWithDropout, or MaxoutWithDropout) Specify the hidden layer dropout ratio to improve generalization. 
#Specify one value per hidden layer. The range is >= 0 to <1, and the default is 0.5.

#l1: Specify the L1 regularization to add stability and improve generalization; sets the value of many weights to 0.
#
#l2: Specify the L2 regularization to add stability and improve generalization; sets the value of many weights to smaller values.

model = H2ODeepLearningEstimator(distribution='Gaussian',
                                   standardize = True,
	                               activation='Rectifier', 
	                               hidden=[20,20,20],
	                               l1=1e-5,
                                   l2=1e-5,
	                               epochs=10,
	                               nfolds=nfolds,
	                               fold_assignment=fold_assignment,
                                   keep_cross_validation_predictions=True)

#             PREDICTIONS      TEST
#PREDICTIONS     1.000000  0.367742
#TEST            0.367742  1.000000
##
#variable    relative_importance    scaled_importance    percentage
#----------  ---------------------  -------------------  ------------
#NVDI        1                      1                    0.324792
#SLOPE       0.368962               0.368962             0.119836
#NDSM        0.312297               0.312297             0.101432
#ROUGHNESS   0.190196               0.190196             0.0617742
#SEQRED      0.18928                0.18928              0.0614766
#SEQGREEN    0.16418                0.16418              0.0533243
#SEQREDEDGE  0.122736               0.122736             0.0398637
#NIR         0.120583               0.120583             0.0391645
#GREEN       0.117622               0.117622             0.0382027
#GNDVI       0.111782               0.111782             0.0363059
#RENVDI      0.10294                0.10294              0.0334341
#BLUE        0.0961154              0.0961154            0.0312175
#TPI         0.0924823              0.0924823            0.0300375
#RED         0.0897192              0.0897192            0.0291401

model.train(y="THERMAL", x=['BLUE', 'GREEN', 'RED', 'SEQGREEN', 'SEQRED', 'SEQREDEDGE', 'NIR','GNDVI', 'NVDI','RENVDI','NDSM','SLOPE','TPI','ROUGHNESS' ],training_frame=train,validation_frame = test)

metrics = model.model_performance()       

print(metrics)

predict = model.predict(test)

predict.head()

pred = model.predict(test[0:-1]).as_data_frame(use_pandas=True)
test_actual = test.as_data_frame(use_pandas=True)['THERMAL']

predictions= pred['predict'].values.reshape(-1, 1)
inversed_pred = THRscaler.inverse_transform(predictions)
print(inversed_pred)

test_actual= test_actual.values.reshape(-1, 1)
inversed = THRscaler.inverse_transform(test_actual)
print(inversed)

res = pd.DataFrame(
    {'PREDICTIONS': inversed_pred.ravel(),
     'TEST': inversed.ravel()
    })

res.to_csv('./PREDICTED_TEST_20_.csv')

ax = plt.gca()

res.plot(kind='line',y='PREDICTIONS', color='red', ax=ax)
res.plot(kind='line',y='TEST',color='green',ax=ax)

plt.show()

data = res[['PREDICTIONS','TEST']]
correlation = data.corr(method='pearson')
print("Corellation")
print(correlation)

print("Covariance")
print(res.cov())

print("Training AUC")
print(model.auc)