from __future__ import division

import glob

import matplotlib.pyplot as plt
import pandas as pd
from numpy import genfromtxt

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

import h2o
from h2o.estimators import H2ODeepLearningEstimator

h2o.init()

h2o.cluster().show_status()

data = h2o.H2OFrame(bands_list)

splits = data.split_frame(ratios=[0.7, 0.15], seed=1)  

train = splits[0]
valid = splits[1]
test  = splits[2]

nfolds = 10
fold_assignment = 'Random'

model = H2ODeepLearningEstimator(distribution='Gaussian',
                                       standardize = True,
	                               activation='Rectifier', 
	                               hidden=[200,200,200],
	                               l1=1e-5,
                                       l2=1e-5,
	                               epochs=10,
	                               nfolds=nfolds,
	                               fold_assignment=fold_assignment,
                                       keep_cross_validation_predictions=True)

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
