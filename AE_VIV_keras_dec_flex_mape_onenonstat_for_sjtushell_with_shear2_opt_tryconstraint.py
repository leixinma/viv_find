# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 13:35:55 2018

@author: leixin
"""
## THIS IS THE ORIGIANL VERSION WITHOUT THE REGULARIZATION
#from matplotlib import pyplot as plt

from __future__ import absolute_import, division, print_function
import pandas as pd
import tensorflow as tf
from keras.regularizers import Regularizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import itertools
import scipy.io as sio
from matplotlib import cm
from tensorflow import keras
import keras
import math
#from scipy.stats import halfnorm
import random
from keras.models import Model,Sequential
from keras.layers import Input,Dense, Dropout, Flatten
from keras import regularizers
import pandas as pd
import keras.backend as K
import matplotlib as mpl
import random
from matplotlib.colors import LinearSegmentedColormap
from numpy.random import seed 
seed(1) 
import os
from random import randint
from matplotlib.ticker import MaxNLocator
#from matplotlib import *
import matplotlib.ticker as ticker
from keras.layers import Input, Dense
from numpy import linalg as LA
from keras import backend as K
from keras import regularizers
import numpy as np
from keras.regularizers import Regularizer
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from keras.layers import Dropout, Flatten
from keras.callbacks import EarlyStopping
from tensorflow.compat.v1 import set_random_seed 
import sys
import AE_VIV_flex_preprocessing as pp
import entropy_estimators as ee
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

#import keras.models import mpl_toolkits
#from FIGPLOTS import *
#from keras.wrappers.scikit_learn import KerasRegressor
set_random_seed(2)
##from sklearn.ensemble import AdaBoostRegressor
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}'] #for \text command    
##
def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

class L21(Regularizer):
    """Regularizer for L21 regularization.
    # Arguments
        C: Float; L21 regularization factor.
    """

    def __init__(self, C=0.):
        self.C = K.cast_to_floatx(C)

    def __call__(self, x):
        const_coeff = np.sqrt(K.int_shape(x)[1])
        return self.C*const_coeff*K.sum(K.sqrt(K.sum(K.square(x), axis=1)))

    def get_config(self):
        return {'C': float(self.C)}

def intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2] 
    return lst3 
def movingaverage(interval, window_size):
    window= np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'valid')

def predict_with_uncertainty(f, x, no_classes, n_iter):
    result = np.zeros((n_iter,) + (x.shape[0], no_classes) )

    for i in range(n_iter):
        result[i,:, :] = f((x, 1))[0]

    prediction = result.mean(axis=0)
    uncertainty = result.std(axis=0)
    return prediction, uncertainty    
def customLoss(yEst,yPred):
    return K.sum(K.log(yEst) - K.log(yPred))
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

@tf.custom_gradient
def grad_reverse(x):
    y = tf.identity(x)
    def custom_grad(dy):
        return -dy
    return y, custom_grad

class GradReverse(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__(name="grl")

    def call(self, x):
        return grad_reverse(x)    

# Display training progress by printing a single dot for each completed epoch
""" ========NOTE: MEANSTATS ARE NO LONGER USED==="""
""" =============MAIN================="""
## 80 mm diameter
""" ========I created params to tune on 3/23==="""
newcheck=0
if newcheck==1:
    curlearn=.8## APPLY CURRICULUM LEARNING## normal training please set below large lims, curlearn=0;
else:
    curlearn=.8#799

pruning=1
#l1rall=0
timeindex=41
#l1rall=[0.001,0.01,0.1,0.2,0.4,0.6,0.8]
l1rall=[0.01,0.1,.2,.25,.28,.3,.4,.45,0.5,.7,1,1.5]
l1rall=[.3,.5,1,1.5,2]
#l1rall=[.12,.13]
l1rall=[.25,.3,.4,.5,1]
l1rall=[.01,.1,.5,1,5]
#l1rall=[0.01]
#l1rall=[3,4]
l1rall=[.45]
l1rall=[0.01,0.1,.12,.13,.15,.2]
l1rall=[0.01,0.1,.15,.2,.25,.28,.3,.4,.45,.5,.7,1]
#l1rall=[.2,.25]
l1rall=[.25,.3,.35,.4]
l1rall=[.01,.05,.1,.2]
l1rall=[0.001]
l1rall=[.001,.01,.02,.03,.05,.1,.2]
l1rall=[.01,.05,.1,.2,.3,.5]
l1rall=[.02,.03,.04,.05]
l1rall=[.015]
l1rall=[.005,.01,.02,.03,.05,.1,.15,.2,.3,.4,.5,.6,.7,.8,.9,1,1.1,1.2]
#l1rall=[.5,.6,.7,.8,.9,1,1.1,1.2,1.4,1.6,1.8]
#l1rall=[.6,.8,1,1.2,1.4,1.6,1.8]
l1rall=[.005,.05,.1,.15,.17,.2,.22,.25]
l1rall=[.3,.35,.4,.45,.5]
l1rall=[.6,.7,.8,.9,1,1.1,1.2,1.3,1.4,1.5]
l1rall=[.1,.2,.4,.6,.8,1,1.2]
#l1rall=[.25,.3,.45,.5]
#l1rall=[1,1.2,1.4,1.6,1.8]
l1rall=[.005,.05,.1,.15,.2,.25,.3,.35,.4]
l1rall=[.05,.1,.15,.2,.25,.3,.35]
l1rall=[.15,.2]
l1rall=[.15,.2,.25,.3,.35]
l1rall=[.16,.17,.18,.19,.2,.21,.22,.23,.24,.25,.3,.35]
tt=.00001*0
if tt==0:
    mapee=1
else:
    mapee=0
mapee=6## wode hahaha2 for c* 3 for c* and re
#l1rall=[.05,.1,.15,.2,.25,.3,.35,.4]
l1rall=[1.5*10**(-6)]#list(np.asarray([.005,.01,.02,.03,.05,.1,.15,.2,.25,.3,.35,.4,.5,.6,.8,1])*(1-tt)*0.001)
l1rall=[0.01,.02,.03,.04,.05,.1]
l1rall=[.005]
    ##-1 pure uniform, 1 represents all 0 represents only uniform 2 represents only sheared 3 no strakes
uuniform=3#-==-1 pure uniform, 1 represents all 0 represents only uniform 2 represents only sheared 3 no strakes

if uuniform==-1:
    l1rall=[.00443]

if uuniform==0:
    l1rall=[.0048,.0098,.0148,.0198,.0248]
    l1rall=[.0448,.098,.148,.198,.248,.298,.36]
    l1rall=[.0448]
    l1rall=[.000448]
if uuniform==2:
    l1rall=[.0447,.097,.147,.197,.247,.297,.37,.447,.497,.547]
    l1rall=[.87]
    l1rall=[.00447]
if uuniform==3:
    l1rall=[.0446,.096,.146,.196,.246,.296,.36,.446,.496,.546]
#    l1rall=[.189]
#    l1rall=[.646,.746,.846,.946]
#    l1rall=[.00446]
#l1rall=list(.99*np.asarray([.05,.1,.15,.2,.25,.3,.35]))
#l1rall=list(.99*np.asarray([.22]))
#l1rall=[.005,.01,.05,.1,.15,.17,.2]
#l1rall=[.005,.01,.05,.1,.15,.17,.2,.25,.3]
l1rall=[.00001,.0001,.001,.01]
l1rall=[.00001,.0001,.001,.0025]
l1rall=list(np.asarray([.00011,.0011,.0021])*1)
l1rall=[.00081]

l1rall=[.0001]
#######start
#l1rall=[.0011,.0015,.002,.0025,.003,.004,.005,.006,.007,.008]
l1rall=[.0015]
#l1rall=[.009]
l1rall=[.0001]
#l1rall=[.01,.015]
l1rall=[.0001]
###end
okk=1
for l1sel in range(0,len(l1rall)):
#        nnnumall=[2]
#    else:
    if mapee==1:
        nnnumall=[2]## for 532 only
    else:
        nnnumall=[2]#3
#    nnnumall=[2]
    randomforestt=0
    if pruning==0:
#        dstep=0.001
#        dropp=0.05
#        nnnumall=[3]
#        l2r=1e-06
        dstep=0.2
        dropp=0.001
        l2r=1e-01
    else:
        dstep=0.799## for curlearn=0
        dropp=0.0001
#        dropp=.001 ## for high
        l2r=1e-01
    paramstotune=[dropp, l2r, 6000,dstep/2,2,19,1] ## good first attempt, full=0 is also good## for new gl1
    paramstotune=[dropp, l2r, 6000,dstep/2,2,19,100] ## include strakes, good first attempt, full=0 is also good## for new gl1
##    paramstotune=[dropp, l2r, 6000,dstep/2,2,19,200] ## include strakes, good first attempt, full=0 is also good## for new gl1
##    paramstotune=[dropp, l2r, 6000,dstep/2,2,19,113] ## include strakes, good first attempt, full=0 is also good## for new gl1
##
#    paramstotune=[dropp, l2r, 3000,dstep,1,30,100] ##include strakes, good first attempt, fUull=0 is also good## for new gl1
#    paramstotune=[dropp, l2r, 6000,dstep/2,2,19,100] ## good first attempt, full=0 is also good## for new gl1
#    paramstotune=[dropp, l2r, 6000,dstep,0,30,100] ##include strakes, good first attempt, full=0 is also good## for new gl1
    paramstotune=[dropp, l2r, 3000,dstep,0,70,700] ##include strakes, good first attempt, full=0 is also good## for new gl1
    paramstotune=[dropp, l2r, 3000,dstep,1,70,700] ##include strakes, good first attempt, full=0 is also good## for new gl1
    paramstotune=[dropp, l2r, 2000,dstep/2,1,80,800] ##include strakes, good first attempt, full=0 is also good## for new gl1
#    paramstotune=[dropp, l2r, 2000,dstep/2,0,80,780] ##include strakes, good first attempt, full=0 is also good## for new gl1
#    paramstotune=[dropp, l2r, 2000,dstep/2,2,40,140] ##include strakes, good first attempt, full=0 is also good## for new gl1

#    paramstotune=[dropp, l2r, 2000,dstep/2,0,80,700] ##include strakes, good first attempt, full=0 is also good## for new gl1
#    paramstotune=[dropp, l2r, 3000©,dstep,1,80,233] ##include strakes, good first attempt, full=0 is also good## for new gl1
    paramstotune=[dropp, l2r, 3000,dstep,0,30,100] ##include strakes, good first attempt, full=0 is also good## for new gl1
    paramstotune=[dropp, l2r, 6000,dstep,0,30,100] ##include strakes, good first attempt, full=0 is also good## for new gl1
    paramstotune=[dropp, l2r, 6000,dstep,0,60,100] ##include strakes, good first attempt, full=0 is also good## for new gl1
    paramstotune=[dropp, l2r, 6000,dstep,0,80,780] ##include strakes, good first attempt, full=0 is also good## for new gl1
#    paramstotune=[dropp, l2r, 3000,dstep,1,80,800] ##include strakes, good first attempt, full=0 is also good## for new gl1
    paramstotune=[dropp, l2r, 3000,dstep,0,80,700] ##include strakes, good first attempt, full=0 is also good## for new gl1
    paramstotune=[dropp, l2r, 3000,dstep,1,80,700] ##include strakes, good first attempt, full=0 is also good## for new gl1
    paramstotune=[dropp, l2r, 3000,dstep,0,80,700] ##include strakes, good first attempt, full=0 is also good## for new gl1
    paramstotune=[dropp, l2r, 3000,dstep,2,30,236] ##include strakes, good first attempt, full=0 is also good## for new gl1
    paramstotune=[dropp, l2r, 6000,dstep,0,30,233] ##include strakes, good first attempt, full=0 is also good## for new gl1
    paramstotune=[dropp, l2r, 6000,.01,2,31,236] ##include strakes, good first attempt, full=0 is also good## for new gl1
    paramstotune=[dropp, l2r, 6000,.01,2,31,140] ##include strakes, good first attempt, full=0 is also good## for new gl1
    paramstotune=[dropp, l2r, 6000,.01,0,30,234] ##include strakes, good first attempt, full=0 is also good## for new gl1
    paramstotune=[dropp, l2r, 6000,.004,2,31,300] ##include strakes, good first attempt, full=0 is also good## for new gl1
#    paramstotune=[dropp, l2r, 6000,.01,1,30,140] ##include strakes, good first attempt, full=0 is also good## for new gl1
#    paramstotune=[dropp, l2r, 6000,.01,1,30,100] ##include strakes, good first attempt, full=0 is also good## for new gl1
    if mapee==1:
        paramstotune=[dropp, l2r, 6000,.02,0,30,710] ##include strakes, good first attempt, full=0 is also good## for new gl1
#        paramstotune=[dropp, l2r, 5000,.01,0,30,235] ##include strakes, good first attempt, full=0 is also good## for new gl1
#        paramstotune=[dropp, l2r, 5000,.01,0,30,233] ##include strakes, good first attempt, full=0 is also good## for new gl1
    else:
        paramstotune=[dropp, .000001, 1000,.2,0,30,100] ##include strakes, good first attempt, full=0 is also good## for new gl1
#    paramstotune=[dropp, l2r, 6000,.01,0,30,710] ##include strakes, good first attempt, full=0 is also good## for new gl1
    paramstotune=[dropp, l2r, 6000,.01,0,30,235]#233] ##include strakes, good first attempt, full=0 is also good## for new gl1
    paramstotune=[dropp, l2r, 6000,.05,0,33,148]#233] ##include strakes, good first attempt, full=0 is also good## for new gl1
    paramstotune=[dropp, l2r, 6000,.05,0,33,142]#233] ##include strakes, good first attempt, full=0 is also good## for new gl1
    paramstotune=[dropp, l2r, 6000,.1,0,33,100]#233] ##include strakes, good first attempt, full=0 is also good## for new gl1
#    paramstotune=[dropp, l2r, 6000,.05,0,33,235]#233] ##include strakes, good first attempt, full=0 is also good## for new gl1
    paramstotune=[dropp, l2r, 6000,.05,0,33,141]#233] ##include strakes, good first attempt, full=0 is also good## for new gl1
    paramstotune=[dropp, l2r, 6000,.05,0,30,152]#233] ##include strakes, good first attempt, full=0 is also good## for new gl1
    paramstotune=[dropp, 0.0000001, 8000,.003,0,30,171]#233] ##include strakes, good first attempt, full=0 is also good## for new gl1
#    paramstotune=[dropp, l2r/5, 3000,.01,0,30,100]#233] ##include strakes, good first attempt, full=0 is also good## for new gl1
#    paramstotune=[dropp/100, 0, 4000,.01,0,30,100]#233] ##include strakes, good first attempt, full=0 is also good## for new gl1
#    paramstotune=[dropp, l2r, 3000,.01,0,30,147]#233] ##include strakes, good first attempt, full=0 is also good## for new gl1
#    paramstotune=[dropp*100, l2r/5, 6000,.1,0,30,100]#233] ##include strakes, good first attempt, full=0 is also good## for new gl1
    print('pruning=',str(pruning))
    #paramstotune=[0.05, 1e-06, 10000,0.0007,1,10,51] ## good first attempt, full=0 is also good
    #paramstotune=[0.05, 5e-05, 11000,0.0006,1,10,49] ## good first attempt, full=0 is also good
    #paramstotune=[0.05, 5e-05, 10000,0.0006,1,10,49] ## good first attempt, full=0 is also good
    #paramstotune=[0.05, 5e-05, 10000,0.0006,1,10,50] ## good first attempt, full=0 is also good    
    
    moveavg=0
    
    """ droprat,regurat,epoch,dt,selectouttrain,uniform or not, full num"""
    import change as c
    testnum30u= c.testnum30u
    testnum30s=c. testnum30s
    testnum40u=c.testnum40u
    testnum40s=c.testnum40s
    testnum50u=c.testnum50u
    testno3=c.testno3
    testnum70u=c.testnum70u
    testnum70s=c.testnum70s
    testnum20s=c.testnum20s
    testnum20u=c.testnum20u
    testnum20f=c.testnum20f
    test70u=c.test70u
    test70s=c.test70s
    test70=c.test70
    teststatid=c.teststatid
    teststatall=c.teststatall
    teststatsomesec=c.teststatsomesec
    testhigh=c.testhigh
#    notscale=1
#    wweeall=[0]
    stremove=0
    notscale=0
    wweeall=[0.000000001]
    wweeall=[0.1,.3,.4,.5,.6,.8]
#    wweeall=[0.001]
    wweeall=[0]
    selecttother=[ 16,9,10,3,  7,14,  19, 20, 34, 38, 39, 40,48,49,50,51]
    selecttother=[0,3, 4, 16, 9, 10, 6, 7,8,14,  15,19, 20,38, 39, 40,48,49,50,51]#1,16, 9, 10, 3, 4, 
    selecttother=[16, 9, 10, 7,8,14,  15,19, 20,38, 39, 40,48,49,50,51]#1,16, 9, 10, 3, 4, 
#    selecttother=[48,49,50,51,1,34,16, 9, 10, 3, 4,  7,14,  19, 20,38, 39, 40,15]#1,16, 9, 10, 3, 4, 
    selecttother=[38,39,15,7,9,10,16,20,40,49,51,3,4,1]#48,50]#49]
#    selecttother=[14,15]
#    selecttother=[49,51]
#    selecttother=[16,19,20,40,51,38,39]
#    selecttother=[19,40,20]
#    selecttother=[0,1,7,8,9,10,14,15,16,38,39,48,50,51]
#    selecttother=[54]
    selecttother=[14]#1,40]
    addadd=0
    if addadd==1:
        xxall=selecttother
    else:
        xxall=wweeall
    nnnum=nnnumall[0]
    for xx in xxall:
        if addadd==1:
            selother=xx
            wwee=wweeall[0]
        else:
            selother=[]
            wwee=xx
        wweet=wweeall[0]+0
        if wwee<=.5 and wwee>.4:
            wweet=.8
        elif wwee==.8:
            wweet=1
#    for nnnum in nnnumall: ## used for sensitivity analysis
        print("NN number",nnnum)
        if wwee==0: 
            traininit=1
######      start Higher harmonics
            paramstotune=[dropp, l2r, 5000,.01,2,30,142] ##include strakes, good first attempt, full=0 is also good## for new gl1
            paramstotune=[dropp, l2r,6000,.006,2,30,142] ##include strakes, good first attempt, full=0 is also good## for new gl1
#            paramstotune=[dropp, l2r,3000,.01,2,30,188] ##include strakes, good first attempt, full=0 is also good## for new gl1

######      end Higher harmonics
#            paramstotune=[dropp, l2r, 3000,.05,1,30,188] ##include strakes, good first attempt, full=0 is also good## for new gl1
            paramstotune=[dropp, l2r,4000,.01,5,30,141] ##include strakes, good first attempt, full=0 is also good## for new gl1
#            paramstotune=[dropp, l2r,3000,.01,5,30,303] ##include strakes, good first attempt, full=0 is also good## for new gl1
#            paramstotune=[dropp, l2r,6000,.01,0,30,234] ##include strakes, good first attempt, full=0 is also good## for new gl1
#            paramstotune=[dropp*1, l2r,3000,.02,0,30,100] ##include strakes, good first attempt, full=0 is also good## for new gl1
            paramstotune=[dropp, l2r, 5000,.03,0,30,189]#include strakes, good first attempt, full=0 is also good## for new gl1
#            paramstotune=[dropp*20, l2r, 5000,.02,0,30,185] ##include strakes, good first attempt, full=0 is also good## for new gl1
#            paramstotune=[dropp*20, l2r, 5000,.02,0,30,234] ##include strakes, good first attempt, full=0 is also good## for new gl1
#            paramstotune=[dropp*20, l2r, 3000,.05,0,30,178] ##include strakes, good first attempt, full=0 is also good## for new gl1
#            paramstotune=[dropp*20, l2r, 5000,.02,0,30,100] ##include sstrakes, good first attempt, full=0 is also good## for new gl1
#            paramstotune=[dropp*20, l2r, 3000,.03,0,30,110] ##include strakes, good first attempt, full=0 is also good## for new gl1
#            paramstotune=[dropp*20, l2r, 3000,.025,0,30,142] ##include strakes, good first attempt, full=0 is also good## for new gl1
#            paramstotune=[dropp*20, l2r, 5000,.03,0,30,142] ##include strakes, good first attempt, full=0 is also good## for new gl1
#            paramstotune=[dropp*400, l2r*10, 3000,.1,1,34,100] ##include strakes, good first attempt, full=0 is also good## for new gl1
            paramstotune=[dropp, l2r, 5000,.03,49,30,190]#include strakes, good first attempt, full=0 is also good## for new gl1

            l1rall[l1sel]=0.0001*1
            l1r=l1rall[l1sel]
        
        else:
            traininit=0
            paramstotune=[dropp/100, 0.0035-1*0.001,4000,.005,0,30,110]#233] importwave=2
#            paramstotune=[dropp/100, 0.003-1*0.001,4000,.005,0,30,142]#233] importwave=3
#            paramstotune=[dropp/100, 0.0027-1*0.001,4000,.005,0,30,189]#233] importwave=3
#            paramstotune=[dropp/100, 0.003-1*0.001,3000,.005,0,30,186]#233] importwave=3
            paramstotune=[dropp/100, 0.0027-1*0.001,4800,.005,0,30,189]#233] importwave=3
#            paramstotune=[dropp/100, 0.0026-1*0.001,4000,.005,0,30,159]#233] importwave=3
#            paramstotune=[dropp/100, 0.003-addadd*0.001,3000,.005,0,3100,171]#233] ##include 50% new
#            paramstotune=[dropp/100, 0.003,4000,.005,0,30,100]#233] ##include 50% new
#            paramstotune=[dropp/100, 0.003-addadd*0.001,3000,.005,0,30,171]#233] ##include 50% new
#            paramstotune=[dropp/100, (0.0028-1*0.001)/1,5000,.005,0,30,100]#233] ##include 50% new
#            paramstotune=[dropp/100, 0.0023-1*0.00,4900,.005,0,30,180]#233] ##include 50% new
            l1r=l1rall[l1sel]

        print('l1=',str(l1r))

        print(str(paramstotune))
        full=paramstotune[6]

        if paramstotune[6]>=600:
            vscale=6##=1 if betaR included for pout regions
            vscalee=6
        else:
            vscale=4##=1 if betaR included for pout regions
            vscalee=vscale+0
        if paramstotune[4]==0:
            vscale=4
            vscalee=vscale+0
        if paramstotune[5]==31:
            vscale=8##=1 if betaR included for pout regions
            vscalee=8
        if paramstotune[6]>900:
            vscale=4##=1 if betaR included for pout regions
            vscalee=4
        if  paramstotune[6]==180 or paramstotune[4]==2:
            vscale=4##=1 if betaR included for pout regions
            vscalee=4
        delett=0
        selectstat=1## 1 CHOSE STATIONARY SECTIONS, ELSE 0
        fixinputdata=1##  1 fix data, 0 random sample in python
        print('data unchaged=',str(fixinputdata))
        plotuncert=1## 1 predict with uncertainty estimates, else0 
        fullinput=1## 1 all features, dont plot trend, else0
        backward=0## backward 1 starting backward elimination backward 2 forward selection (works with fullinput==0)
        bareana=1# 1 for bare section, 0 for damped sections
        onepipe=0  ## DATA from 1 pipe or not
        restrictvr=0 ## constrain the range of vr to 5-7.5
        prunedata=0
        importshear=0  ## 0 FOR SJTU data; 1-3 for shell data
        savemodel=1
        importuniform=paramstotune[5] ## 1--UNIFORM ONLY 2-- BOTH 0-- SHEARED
        if importuniform==40:
            importwave=4## top 30%  control which to dir to read from 
        elif importuniform==50:
            importwave=5
        elif importuniform==60:
            importwave=6
        else:
            importwave=2
        ##otherwise curlearn=1 then nonstat
        print('Curilearn',str(curlearn))
        
#        lims=[.1,.1,.2,.2]
        whtiter=1
        linall=[.3,.3,.4,.25,.35]
        print('IMportwave',str(importwave))
    #    predictthree=0 ## not used
        plotdetail=0
        if paramstotune[5]==7 or paramstotune[5]==6 or paramstotune[5]==8:
            multrat=2
        else:
            multrat=1
        [testnumall,testnumprint,readstring]=pp.testcombine(paramstotune,importuniform,importwave,testnum30u,testnum30s,testnum40u,testnum40s,testnum50u, testnum20u,testnum20s,testnum20f,testnum70u,testnum70s,testnum20f,testnum70u,testnum70s)
        if testnumall[0]==303:
            EPOCHS = 16000
        else:
            EPOCHS =paramstotune[2]               
        
        if paramstotune[4]==2:
            for ii in range(0,len(testno3)):
                testexp=testno3[ii]
                findrow=np.argwhere(testnumall==testexp)
                testnumall=np.delete(testnumall, findrow, axis = 0)
        
        nreduce=2        
#        sys.exit()
        if fixinputdata==0:
            
            [inputt,traindata,testdata]=pp.firstdata0(testnumall,importwave,selectstat,restrictvr,readstring,paramstotune,testnumprint,linall,timeindex)
                    
        else:
            scale2=1
#            train=sio.loadmat('traindata'+str(testnumprint)+'.mat')  
#            test=sio.loadmat('testdata'+str(testnumprint)+'.mat') 
            if xx==xxall[0]:                
                
                if importwave==2 and vscale==4:
                    inputt=sio.loadmat('dataall'+str(testnumprint)+'_ok.mat') 
                    print('mcycle,3')
                elif importwave==2 and vscale==5:
                    inputt=sio.loadmat('dataall'+str(testnumprint)+'_strrat_v0.mat') 
                    print('mcycle,3strrat')
                else:
                    inputt=sio.loadmat('dataall'+str(testnumprint)+'_6cycle.mat') 
                    print('mcycle,6')
                inputt=np.real(inputt['dataall'])
#                sys.exit()
#                if paramstotune[6]==100:
                if inputt.shape[1]>48:
#                    inputt[:,3]=inputt[:,51]
#                    inputt[:,4]=inputt[:,49]*inputt[:,40]
#                    if wwee!=0.4 and wwee!=0 and wwee<.5:
#                        print('rep')
#                        inputt[:,49]=inputt[:,4]
#                        inputt[:,51]=inputt[:,3]
                    if paramstotune[4]==50:
                        inputt[:,50]=inputt[:,0]*inputt[:,14]
                    inputtadd=np.zeros([inputt.shape[0],10])
                    inputt=np.hstack((inputt,inputtadd))
                    inputt[:,54]=inputt[:,52]
                    inputt[:,52]=inputt[:,34]*inputt[:,38]
                    inputt[:,53]=inputt[:,34]*inputt[:,38]*inputt[:,40]
                else:
                    inputt=np.hstack((inputt,inputt[:,0:10]))
                    inputt[:,51]=inputt[:,3]
                    inputt[:,49]=inputt[:,4]
                if paramstotune[4]==5:
                    inputt[:,51]=1./np.sqrt(1+inputt[:,51])
                if paramstotune[4]==5 and scale2==1 and vscale==4:
                    inputt[:,5]=inputt[:,2]/inputt[:,0]
#                if paramstotune[4]==0 and paramstotune[6]==128:
#                    inputt[:,34]=inputt[:,34]*inputt[:,38]
                minn=np.min(np.min(inputt[:,paramstotune[4]]))
                print(str(minn))
                if testnumprint<533:
                    findrow=np.argwhere((inputt[:,0]<=.3))
                    inputt=np.delete(inputt, findrow, axis = 0)
                print([len(np.argwhere((inputt[:,13]>2100) & (inputt[:,13]<2200))),len(np.argwhere((inputt[:,13]>2200) & (inputt[:,13]<2300))),len(np.argwhere((inputt[:,13]>2300) & (inputt[:,13]<3000))),len(np.argwhere((inputt[:,13]>3000) & (inputt[:,13]<3100))),len(np.argwhere((inputt[:,13]>3100) & (inputt[:,13]<3200))),len(np.argwhere((inputt[:,13]>4000) & (inputt[:,13]<4100))),len(np.argwhere((inputt[:,13]>4100) & (inputt[:,13]<5000))),len(np.argwhere((inputt[:,13]>5000) ))])
                stringsave='1'
                selectrainout=[paramstotune[4]]
                splitt0=0.7
                    
                tt0size=4000
                trsize=int(tt0size*splitt0/(1-splitt0))   
                [traindataall,traindatanon,traindatastat,testdataall,testdatanon,testdatastat,eesort,eesort1,eesort2,entropall,entropallm,thres,trainlabelall,testlabelall,ttsize,trsize,testdataold,traindataold]=pp.filtdata(selectrainout,inputt,testnumprint,uuniform,stringsave,full,curlearn,paramstotune,0,importwave,tt0size,mapee)       
    
                [traindata,traindatanon,traindatastat,testdata,testdatanon,testdatastat,eesort,eesort1,eesort2,entropall,entropallm,thres,trainlabel,testlabel,ttsize,trsize,testdataold,traindataold]=pp.filtdata(selectrainout,inputt,testnumprint,uuniform,stringsave,full,curlearn,paramstotune,traininit,importwave,tt0size,mapee)       
                
                minn=np.min(np.min(traindata[:,paramstotune[4]]))
                print('minn',str(minn))
                if traininit==1:
                    traindata[traindata.shape[0]-5:traindata.shape[0],:]=traindata[0:5,:]
                    testdata[testdata.shape[0]-5:testdata.shape[0],:]=testdata[0:5,:]
    #            traindata[:,paramstotune[4]]=(traindata[:,paramstotune[4]])+minn
    #            testdata[:,paramstotune[4]]=(testdata[:,paramstotune[4]])+minn
    #            sio.savemat('test'+str(testnumprint)+str(selectrainout[0])+'u'+str(uuniform)+'.mat', {'testdata':testdata})                
    ##            sio.savemat('train'+str(testnumprint)+str(selectrainout[0])+'u'+str(uuniform)+'.mat', {'traindata':traindata})                
    #            importdataaa=0
    #            if importdataaa==1:
    #                traindata=sio.loadmat('train'+str(testnumprint)+str(selectrainout[0])+'u'+str(uuniform)+'.mat')
    #                testdata=sio.loadmat('test'+str(testnumprint)+str(selectrainout[0])+'u'+str(uuniform)+'.mat')
    #                traindata=traindata['traindata']
    #                testdata=testdata['testdata']
                inputt=np.vstack((traindata,testdata))           
                uniquerows2=np.unique(inputt[:,13])
                print('Number of test cases'+str(uniquerows2)+str(len(uniquerows2)))
                    
#                print('Traindata',traindata.shape)        
#                print('Input data',inputt.shape)    
                aa=traindata
#                print([len(np.argwhere((aa[:,13]>2100) & (aa[:,13]<2200))),len(np.argwhere((aa[:,13]>2200) & (aa[:,13]<2300))),len(np.argwhere((aa[:,13]>2300) & (aa[:,13]<2400))),len(np.argwhere((aa[:,13]>3000) & (aa[:,13]<3100))),len(np.argwhere((aa[:,13]>3100) & (aa[:,13]<4000))),len(np.argwhere((aa[:,13]>4000) & (aa[:,13]<4100))),len(np.argwhere((aa[:,13]>4100) & (aa[:,13]<5000))),len(np.argwhere((aa[:,13]>5000) ))])
        #        sio.savemat('dataall22'+str(testnumprint)+'.mat', {'inputt':inputt})                                          
        #        sys.exit()
                """ ======A please only normalize output I will get really confused========="""    
                minn=np.min(np.min(inputt[:,paramstotune[4]]))
                if minn>0:
                    minn=minn*0.95
                else:
                    minn=minn*1.05
                if selectrainout[0]==0 and minn>.25:
                    minn=.28
                if selectrainout[0]==1:
                    minn=.02
                if selectrainout[0]==2:
                    minn=minn/.95*.98
                print('min=',str(minn))
                print('mintest=',str(np.min(testdata[0:ttsize-1,paramstotune[4]])))
                
                inputt[:,paramstotune[4]]=(inputt[:,paramstotune[4]])-minn
                traindata[:,paramstotune[4]]=(traindata[:,paramstotune[4]])-minn
                testdata[:,paramstotune[4]]=(testdata[:,paramstotune[4]])-minn
        print('Traindata',traindata.shape)        
        print('Input data',inputt.shape)    
        aa=traindata
        print('traindiv',[len(np.argwhere((aa[:,13]>2100) & (aa[:,13]<2200))),len(np.argwhere((aa[:,13]>2200) & (aa[:,13]<2300))),len(np.argwhere((aa[:,13]>2300) & (aa[:,13]<2400))),len(np.argwhere((aa[:,13]>3000) & (aa[:,13]<3100))),len(np.argwhere((aa[:,13]>3100) & (aa[:,13]<4000))),len(np.argwhere((aa[:,13]>4000) & (aa[:,13]<4100))),len(np.argwhere((aa[:,13]>4100) & (aa[:,13]<5000))),len(np.argwhere((aa[:,13]>5000) ))])



        """ DROP WEIGHTS FOR THE FIXED STRUCTURE"""
        n = len(inputt) ## SIZE OF ALL DATA
        selectrainout= [paramstotune[4]] ## =2 for total power coefficient
        sscale=[]
        correct=0
        onepipe=0
        [rscale,selecttall0,inputt_descrp,ref]=pp.prep(pruning,vscale,selectrainout,inputt,full)                
        noutput=len(selectrainout) ## TARGET OUTPUT DATA
        mask = np.ones(len(selecttall0), dtype=bool)
        for ii in range(0,len(selectrainout)):
            whselectrainout=np.argwhere(np.array(selecttall0[:])==selectrainout[ii])  
            mask[whselectrainout] = False
        selecttall=np.array(selecttall0)[mask].tolist()
        for ii in np.concatenate((selectrainout,np.array(selecttall))):
            if ii==selectrainout:
                input_descrp_totab=inputt_descrp[ii]
            else:
                input_descrp_totab=np.vstack((input_descrp_totab,inputt_descrp[ii]))    
           
        if fullinput==1 and backward==0 and addadd>0:
            if selother>=0:
                selectt=selecttall+[selother]
            else:
                selectt=selecttall
            countiter=1
            print('selectt',str(selectt))
        elif fullinput==1 and backward==0 and addadd==0:
            selectt=selecttall
            countiter=1
            print('selectt',str(selectt))
        elif fullinput==1 and backward==1:
            selectt=selecttall
            rest0=sorted(set(selecttall)-set(rscale)-set([selectrainout[0]]))
            rest=list(itertools.combinations(rest0,1))
            
            countiter=len(rest)
            print('countiter',str(countiter))
        elif fullinput==1 and backward==2:
            selectt0=selecttall
            selecttallall=sorted(set(selecttallall0)-set(selecttall0)-set([selectrainout[0]]))
#            selecttother=selecttallall
            selecttother0=[e for e in selecttallall if e not in selectt0] 
            selecttother=list(itertools.combinations(selecttother0,1))
#            selecttother=selecttother[90]
#            b= [x[0] for x in selecttother]
#            for ii in range(0,len(selecttallall)):
#                if selecttother[ii] in selectt0:
#                    selecttother=np.delete(selecttother, ii, axis = 0)
#            selecttother=[1,7,14,15,19,20]
            countiter=len(selecttother)
#            selecttother=[]
            print('countiter',str(countiter))
        elif fullinput==0 and backward==0:
        #    selectt=[10,12,13,18]
            selectt=[10]    
            countiter=1    

        mapeall=np.zeros(countiter)
        mapetrainall=np.zeros(countiter)
        trainout_data=traindata[:,selectrainout]
        testout_data=testdata[:,selectrainout]
        countitert=0
        if pruning==1:
            iterrmax=2
        else:
            iterrmax=1
            
        iterr=0
        print('add. select vars'+str(countiter))
        print('selectt',str(selectt))
        if delett==1:
           xtall=range(countiter)
        else:
           xtall=range(0,iterrmax)
           xtall=range(0,1)
        for xta in xtall:      ## turn either this for on or the for on below    
#        for iterr in range(0,iterrmax):
            if delett==1:
               countitert=xta
            else:
               iterr=xta
        #    selectt=[1,11,12,13]
            if iterr==0:
                 selectt=selectt
                 vscale=vscalee#1  ## REDEFINED VSCALE
                 l1r=l1rall[l1sel]
                 savemodel=0
            else:
                 selectt=selecttnew
                 vscale=0 ## REDEFINED VSCALE
                 if mape==1:
                     l1r=0.005
                 else:
                     l1r=10**(-8)
                 savemodel=1
            if mapee>=2:
                savemodel=1
            if fullinput==1 and backward==1:
                selectt=np.delete(selecttall,countitert, axis = 0)
            if fullinput==0 and backward==2:
                selectt=selecttall0[countitert+1]
            if fullinput==1 and backward==2:
                selectt=list(np.hstack((selectt0,selecttother[countitert])))
#                selectt=selectt0
            train_data_full=traindata+0
            test_data_full=testdata+0
            train_data=traindata[:,selectt]
            test_data=testdata[:,selectt]
            test_dataall=testdataall[:,selectt]
            inputto=np.vstack((traindata[0:trsize,:],testdata[0:ttsize,:]))           
            meanstats=np.zeros((inputto.shape[1],2))
            for ii in range(0,inputto.shape[1]):
                meanstats[ii,0]=np.mean(inputto[:,ii])
                if abs(np.std(inputto[:,ii])) <0.001:# and abs(np.mean(inputt[:,ii]))>0.01:
                    meanstats[ii,1]=1
                else:
                    meanstats[ii,1]=np.std(inputto[:,ii])            
    #        else:
    #            meanstats=np.zeros((inputt.shape[1],2))
    #            meanstats[:,1]=1
    #            meanstats[10,1]=10**5        
    #            meanstats[18,1]=6
            cscale=[34,37,38,39,40,41,44,45,46,47]
            [inputt_data_normin, nscale]=pp.nondimdata(inputt[:,selectt],10,inputto[:,selectt],meanstats[selectt,0],meanstats[selectt,1],cscale,selectt,vscalee,rscale,pruning)
            [train_data_normin, nscale]=pp.nondimdata(train_data,10,train_data[0:trsize,:],meanstats[selectt,0],meanstats[selectt,1],cscale,selectt,vscalee,rscale,pruning)
            [test_data_normin, nscale]=pp.nondimdata(test_data,10,test_data[0:ttsize,:],meanstats[selectt,0],meanstats[selectt,1],cscale,selectt,vscalee,rscale,pruning)
            [train_data_normin0, nscale]=pp.nondimdata(train_data,10,train_data[0:trsize,:],meanstats[selectt,0],meanstats[selectt,1],cscale,selectt,vscalee,rscale,pruning)
            [test_data_norminall, nscale]=pp.nondimdata(test_dataall,10,test_data[0:ttsize,:],meanstats[selectt,0],meanstats[selectt,1],cscale,selectt,vscalee,rscale,pruning)

            if fixinputdata==1 and iterr==0:
                [train_data_stat_normin, nscale]=pp.nondimdata(traindatastat[:,selectt],1,test_data_normin,meanstats[selectt,0],meanstats[selectt,1],cscale,selectt,vscalee,rscale,pruning)
                [train_data_non_normin, nscale]=pp.nondimdata(traindatanon[:,selectt],1,test_data_normin,meanstats[selectt,0],meanstats[selectt,1],cscale,selectt,vscalee,rscale,pruning)
                [test_data_stat_normin, nscale]=pp.nondimdata(testdatastat[:,selectt],1,test_data_normin,meanstats[selectt,0],meanstats[selectt,1],cscale,selectt,vscalee,rscale,pruning)
                [test_data_non_normin, nscale]=pp.nondimdata(testdatanon[:,selectt],1,test_data_normin,meanstats[selectt,0],meanstats[selectt,1],cscale,selectt,vscalee,rscale,pruning)
            print("Selected variable:{}".format(selectt))        
            print("Training set: {}".format(train_data.shape))  #eg 404 examples, 13 features
            print("Testing set:  {}".format(test_data.shape))   #eg 102 examples, 13 features
## First store the solution be
            test_data_norminorig=test_data_normin*1
## Then import new solution            
            if len(l1rall)>0 and vscale>=2 and pruning==1:
                if l1r<=0:
                    nscale=10
#                elif vscale==5:
#                    nscale=10
                else:
                    nscale=10
                if mapee>=2 and notscale==1:
                    nscale=50
                if mapee>=2 and notscale==0 and stremove==1:
                    nscale=100
                elif mapee>=2 and notscale==0 and stremove==0:
                    nscale=1
                for rr in range(0,len(selectt)):    
                    if selectt[rr] in rscale:
                        inputt_data_normin[:,rr]=nscale*inputt_data_normin[:,rr]
                        train_data_normin[:,rr]=nscale*train_data_normin[:,rr]
                        test_data_normin[:,rr]=nscale*test_data_normin[:,rr]            
            
            print('rscale',str(rscale))
            print('nscale',str(nscale))
            
            """ TRAINING NN"""
            if randomforestt==1:
                selectt=[6,8,34,10,19]
                X=traindata[:,selectt]
#                selecttall0=[1,51,49,6,7,8,9,10,14,15,16,19,20,34,38,39,40]  
#                X=traindata[:,selecttall0]
                y=traindata[:,selectrainout]
                reg = LinearRegression().fit(X, y)
                scores=reg.score(X, y)
                print('scores',str(scores))
                coeffs=reg.coef_
                scores = cross_val_score(reg, X,y, cv=5)   
                print('scores',str(scores))
                print('coeff',coeffs)         
                
                rf = RandomForestRegressor(n_estimators=50, random_state=20)
                rf.fit(train_data_normin, traindata[:,selectrainout])
                print("Features sorted by their score:")
                print(sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_),input_descrp_totab[1:]), reverse=True))
                print(rf.score(train_data_normin, traindata[:,selectrainout]))
                print(rf.score(test_data_normin, testdata[:,selectrainout]))
                y_pred = rf.predict(test_data_normin)
                y_predtr=rf.predict(train_data_normin)
                fig = plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
                ax = fig.add_subplot(111)
                plt.xlabel('Measured output', fontsize = 16)
                plt.ylabel('Predicted output', fontsize = 16)
                plt.plot(testdata[:,selectrainout]+minn,y_pred+minn,'b*')
                plt.xticks(fontsize=16)  
                plt.yticks(fontsize=16)
                plt.grid()
                plt.rc('font',family='Times New Roman')        
                print('mapetrain',str(mape(traindata[:,selectrainout], y_predtr)))
                print('mapetest',str(mape(testdata[:,selectrainout], y_pred)))
                
                importances = rf.feature_importances_
                importances['random_forest'] = rf.feature_importances_
#                criteria = criteria + ('random_forest',)
                idx = 1
                sys.exit()
            if selectrainout[0]!=2:
                vscalee=vscale+0
                vscalee0=vscale+0
            else:
                vscalee0=24
            """ TRAINING NN"""
            if importwave==0:
                stringsavemod='model'+str(testnumprint)+str(full)+'out'+str(selectrainout[0])
            elif importwave>0 and pruning==1:
                curlearnn=curlearn+0
                stringsavemod='model'+str(testnumprint)+str(full)+'wave'+str(importwave)+'out'+str(selectrainout[0])+'l1r'+str(l1r)+'vscale'+str(vscalee)+'curlearn'+str(curlearnn)
                stringsavemod0='model'+str(testnumprint)+str(full)+'wave'+str(importwave)+'out'+str(selectrainout[0])+'l1r'+str(0.001)+'vscale'+str(vscalee0)+'curlearn'+str(curlearnn)

            else:
                stringsavemod='model'+str(testnumprint)+str(full)+'wave'+str(importwave)+'out'+str(selectrainout[0])+'vscale'+str(vscale)+'curlearn'+str(curlearn)
            
            if full>=999:
                    stringsavemod='model'+str(testnumprint)+str(full)+'wave'+str(importwave)+'out'+str(selectrainout[0])+'l1r'+str(l1r)+'vscale'+str(4)+'curlearn'+str(curlearn)+'u'+str(uuniform)
                    stringsavemod0='model'+str(testnumprint)+str(full)+'wave'+str(importwave)+'out'+str(selectrainout[0])+'l1r'+str(0.001)+'vscale'+str(4)+'curlearn'+str(curlearn)+'u'+str(uuniform)

            else:
                curlearnn=curlearn+0
#                curlearnn=.8
                    
                if uuniform!=1 and paramstotune[6]!=142:
                    stringsavemod='model'+str(testnumprint)+str(full)+'wave'+str(importwave)+'out'+str(selectrainout[0])+'l1r'+str(l1r)+'vscale'+str(vscalee)+'curlearn'+str(curlearnn)+'u'+str(uuniform)
                    stringsavemod0='model'+str(testnumprint)+str(full)+'wave'+str(importwave)+'out'+str(selectrainout[0])+'l1r'+str(0.001)+'vscale'+str(vscalee)+'curlearn'+str(curlearnn)+'u'+str(uuniform)
                elif uuniform!=1 and paramstotune[6]==142:
                    stringsavemod='model'+str(testnumprint)+str(full)+'wave'+str(importwave)+'out'+str(selectrainout[0])+'l1r'+str(l1r)+'vscale'+str(vscalee)+'curlearn'+str(curlearnn)+'u'+str(uuniform)
                    stringsavemod0='model'+str(testnumprint)+str(full)+'wave'+str(importwave)+'out'+str(selectrainout[0])+'l1r'+str(0.001)+'vscale'+str(vscalee0)+'curlearn'+str(curlearnn)+'u'+str(uuniform)

            stringsavemodorig=stringsavemod+'mape'+str(mapee)+'ww'+str(0)
            stringsavemodorig0=stringsavemod0+'mape'+str(mapee)+'ww'+str(0)
            
            if mapee>=2:
                stringsavemod=stringsavemod+'mape'+str(mapee)+'ww'+str(wwee)
                
                
            if addadd>0:
                stringsavemod=stringsavemod+str(selother)
                stringsavemodorig=stringsavemodorig+str(selother)
                stringsavemodorig0=stringsavemodorig0+str(selother)
            [model,droprat,regurat,layernums,l1r] = pp.build_modelmapee(noutput,nnnum,paramstotune,l1r,pruning,train_data,mapee,thres,wwee,traininit)
            model.summary()   
            # Store training stats
            splitt=0.1
            batchsize=128#*2*2*4*4
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=1000)
            if mapee<2:
                histories = model.fit([train_data_normin,trainlabel], trainout_data, batch_size=batchsize, epochs=EPOCHS,validation_split=splitt, verbose=0,callbacks=[es])
                [mapetrainout]=pp.plot_history(histories,splitt,mapee)
            else:
                if traininit==1:
                    xb=train_data_normin[0:trsize,:]
                    xb0=trainlabel[0:trsize,:]
                    yb=trainout_data[0:trsize,:]
                    histories = model.fit([xb,xb0], yb,  batch_size=batchsize, epochs=int(EPOCHS),
                                    validation_split=splitt, shuffle=True, verbose=0)
                    [mapetrainout]=pp.plot_history(histories,splitt,mapee)
                    train_predictions = model.predict([train_data_normin[0:trsize,:],trainlabel[0:trsize,:]])
                    print(str(train_predictions))
                    tend=int(ttsize+(test_data.shape[0]-ttsize)/2)
                else:
                    inp=sio.loadmat(stringsavemodorig0+'.mat')
                    inp=inp['inp'][0]
                    l1rinp=inp[3][0][0]
                    [loadmodel,dropratload,reguratload,layernumsload,l1rload] = pp.build_modelmapee(inp[0][0][0],inp[1][0][0],inp[2][0],l1rinp,inp[4][0][0],inp[5],inp[6][0][0],inp[7][0][0],0,traininit)
                    xb=train_data_normin
                    xb0=trainlabel
                    yb=trainout_data
                    splitt=0.1
                    batchsize=128*4*4*4                    
#                    eee=test_predictions[ttsize:tend,:]-testout_data[ttsize:tend,:]
#                    pk=np.argwhere(eee>0).T[0]
#                    rrr=len(pk)/(test_data.shape[0]-ttsize)/2
#                    print('rrr',str(rrr))
                    histories = model.fit([xb,xb0], yb,  batch_size=batchsize, epochs=1,validation_split=splitt, shuffle=True, verbose=0,callbacks=[PrintDot()])
                    loadmodel.load_weights(stringsavemodorig0+'.h5')
                    ll=0
                    for layer in loadmodel.layers:
                        weights = layer.get_weights() # list of numpy arrays       
                        model.layers[ll].set_weights(weights)
                        loadmodel.layers[ll].set_weights(weights)
                        ll=ll+1
                        
                    if nscale>1:
                        weightscaled=loadmodel.layers[1].get_weights()*1
                        for ii in range(0,len(rscale)):
                            nn=np.argwhere(np.asarray(selectt)==rscale[ii])[0][0]
                            weightscaled[0][nn]=weightscaled[0][nn]/nscale
                            print(rscale[ii])
                        model.layers[1].set_weights(weightscaled)
                        
#                    model.set_weights(Wsave)
#                    weights = layer.get_weights() # list of numpy arrays
                    tend=int(ttsize+(test_data.shape[0]-ttsize)/2)
                    histories = model.fit([xb,xb0], yb,  batch_size=batchsize, epochs=0,validation_split=splitt, shuffle=True, verbose=0,callbacks=[PrintDot()])
                    test_predictions = model.predict([test_data_normin,testlabel])                
                    [losstrendold,losstrendold1]=pp.losstrendv2(testout_data[ttsize:test_data.shape[0],:],test_predictions[ttsize:test_data.shape[0],:],testlabel[ttsize:test_data.shape[0],:],wwee)
                    if l1sel==0:
                        errormapeold=np.mean(abs(test_predictions[0:ttsize-1]-(testout_data[0:ttsize-1]))/(testout_data[0:ttsize-1]))*100
                        print('errorold',str(errormapeold))
                        print('old',str(losstrendold))
                        print('old1',str(losstrendold1))
    #                    sys.exit()
                        bdivedold=[]
                        infold=[]
                        for ii in range(0,4):
                            if ii==0:
                                bu=np.argwhere(testdata[:,40]==1)  
                            elif ii==1:
                                bu=np.argwhere(testdata[:,40]<1)     
                                
                            elif ii==2:
                                bu=np.argwhere((testdata[:,4]>9)) 
                            elif ii==3:
                                bu=np.argwhere((testdata[:,4]>9) &(testdata[:,34]>0)) 
                            if len(bu)>0:
                                buc=bu[(bu<tend) & (bu>ttsize)]
                                bur=bu[bu>tend]
                                difff=np.abs(np.sign(np.abs(testlabel)-thres)).flatten()
#                                sys.exit()
                                cee=np.sum(np.sign(test_predictions[buc].flatten()-(testout_data[buc].flatten()))+1)/2/len(buc)
                                ree=1-np.sum(np.sign(test_predictions[bur].flatten()-(testout_data[bur].flatten()))+1)/2/len(bur)
#                                cee1=np.sum((1-difff[buc])*(np.sign(test_predictions[buc].flatten()-(testout_data[buc].flatten()))+1))/2/len(buc)
#                                ree1=np.sum((1-difff[bur])*(np.sign(-test_predictions[bur].flatten()+(testout_data[bur].flatten()))+1))/2/len(bur)
#                                cee2=np.sum((difff[buc])*(np.sign(-test_predictions[buc].flatten()+(testout_data[buc].flatten()))+1))/2/len(buc)
#                                ree2=np.sum((difff[bur])*(np.sign(test_predictions[bur].flatten()-(testout_data[bur].flatten()))+1))/2/len(bur)
#                                cee=cee1+cee2
#                                ree=ree1+ree2
                                if ii<=1:
                                    bdivedold=bdivedold+[len(buc)/(tend-ttsize),cee,ree]
                                elif ii>=2:
                                    bue=bu[(bu<ttsize)]
                                    errormapeold0=np.mean(abs(test_predictions[bue]-(testout_data[bue]))/(testout_data[bue]))*100
                                    infold=infold+[errormapeold0,cee,ree]
                            elif len(bu)==0 and uuniform<5:
                                bdivedold=bdivedold+[0,0,0]
                        print('divideloss',str(bdivedold))
                        if len(np.argwhere(testdata[0:ttsize-1,34]>0))>0:
                            findrow=np.argwhere(testdata[0:ttsize-1,34]>0).T[0]
                            errormapesold=np.mean(np.abs(testout_data[findrow]-test_predictions[findrow])/testout_data[findrow])
            #                plt.plot(preds, testdata[findrow,selectrainout],'*')
                            print("SHEARED FLOW Testing set Mean Abs Error: {:10.5f}".format(errormapesold))
                        else:
                            errormapesold=0
                        if len(np.argwhere(testdata[0:ttsize-1,34]==0))>0:
                            findrow=np.argwhere(testdata[0:ttsize-1,34]==0).T[0]
                            errormapeuold=np.mean(np.abs(testout_data[findrow]-test_predictions[findrow])/testout_data[findrow])
                            print("UNIFORM FLOW Testing set Mean Abs Error: {:10.5f}".format(errormapeuold))
                        else:
                            errormapeuold=0
                    histories = model.fit([xb,xb0], yb,  batch_size=batchsize, epochs=1,validation_split=splitt, shuffle=True, verbose=0,callbacks=[PrintDot()])
#                    eee=test_predictions[t,:]-testout_data[ttsize:tend,:]
#                    pk=np.argwhere(eee>0).flatten()
#                    rrr=len(pk)/(test_data.shape[0]-ttsize)/2
#                    print('rrr',str(rrr))
#                    sys.exit()
#                    [losstrendold,losstrendold1,mm2,mm3]=pp.losstrendv2(testout_data[ttsize:tend,:],test_predictions[ttsize:tend,:],testlabel[ttsize:tend,:],wwee)
#                    plt.plot(testout_data[ttsize:tend,:],test_predictions[ttsize:tend,:],'o')
#                    [losstrendold,losstrendold1]=pp.losstrendv2(testout_data[tend:test_data.shape[0],:],test_predictions[tend:test_data.shape[0],:],testlabel[tend:test_data.shape[0],:],wwee)
#                    plt.plot(testout_data[tend:-1,:],test_predictions[tend:-1,:],'o',testout_data[0:ttsize,:],test_predictions[0:ttsize,:],'s')
#                    print('old',str(losstrendold))
#                    print('old1',str(losstrendold1))
                    xb=train_data_normin
                    xb0=trainlabel
                    yb=trainout_data
                    splitt=0.1
                    batchsize=128*4*4*2
                    print('batchrate',str(batchsize/traindata.shape[0]))
                    histories = model.fit([xb,xb0], yb,  batch_size=batchsize, epochs=int(EPOCHS),
                                    validation_split=splitt, shuffle=True, verbose=0)#callbacks=[es]
                    [mapetrainout]=pp.plot_history(histories,splitt,mapee)
    
                    mapetrainall[countitert]=np.array(mapetrainout)
                    train_predictions = model.predict([train_data_normin,trainlabel])
                    test_predictions = model.predict([test_data_normin,testlabel])
#            inputt_predictions=model.predict(inputt_data_normin) ## FOR train+ test data
            sell=paramstotune[4]
            entp1=ee.histt(traindatastat[:,sell],0,1) 
            entp2=ee.histt(traindatanon[:,sell],0,1) 
            entp3=ee.histt(traindata[:,sell],0,1) 

            if mapee>1 and traininit==0:
#                (losstrend*4.2*.9+errormape*.1/100)/5.2
                mape = model.evaluate([test_data_normin,testlabel], testout_data, verbose=0)
                mapetrain=model.evaluate([train_data_normin,trainlabel], trainout_data, verbose=0)
            else:
                mape = model.evaluate([test_data_normin,testlabel], testout_data, verbose=0)[1]
                mapetrain=model.evaluate([train_data_normin,trainlabel], trainout_data, verbose=0)[1]
            print("Testing set loss: {:7.2f}".format(mape))
            print("Training set loss: {:7.2f}".format(mapetrain))
            if mapee>1 and wwee>0:
                test_predictions = model.predict([test_data_normin,testlabel])
#                errormape=np.mean(abs(test_predictions[0:ttsize-1].flatten()-(testout_data[0:ttsize-1].flatten()))/(testout_data[0:ttsize-1].flatten()))*100
                errormape=np.mean(abs(test_predictions[0:ttsize-1]-(testout_data[0:ttsize-1]))/(testout_data[0:ttsize-1]))*100
                alosstrend= model.evaluate([test_data_normin[0:ttsize,:],testlabel[0:ttsize]], testdata[0:ttsize,selectrainout[0]])

                errormapetr=np.mean(abs(train_predictions[0:trsize-1]-(trainout_data[0:trsize-1]))/(trainout_data[0:trsize-1]))*100

                blosstrend= model.evaluate([test_data_normin[ttsize:test_data.shape[0],:],testlabel[ttsize:test_data.shape[0]]], testdata[ttsize:test_data.shape[0],selectrainout[0]])
#                losstrendtr= model.evaluate([train_data_normin[trsize:train_data.shape[0],:],trainlabel[trsize:train_data.shape[0]]], traindata[trsize:train_data.shape[0],selectrainout[0]])

                tend=int(ttsize+(test_data.shape[0]-ttsize)/2)
                losstrend000 = model.evaluate([train_data_normin[:,:],trainlabel], traindata[:,selectrainout[0]])

                losstrend000 = model.evaluate([test_data_normin[:,:],testlabel], testdata[:,selectrainout[0]])
                losstrend00 = model.evaluate([test_data_normin[0:ttsize,:],testlabel[0:ttsize]], testdata[0:ttsize,selectrainout[0]])
                losstrend10= model.evaluate([test_data_normin[ttsize:tend,:],testlabel[ttsize:tend]], testdata[ttsize:tend,selectrainout[0]])
                losstrend20 = model.evaluate([test_data_normin[tend:-1,:],testlabel[tend:-1]], testdata[tend:-1,selectrainout[0]])
                losseset0=losstrend00 /5+(losstrend10*2+losstrend20*2)/5-losstrend000
                [losstrend100,losstrend1]=pp.losstrendv2(testout_data[ttsize:tend,0],test_predictions[ttsize:tend,0],testlabel[ttsize:tend,0],wwee)
                [losstrend200,losstrend1]=pp.losstrendv2(testout_data[tend:-1,0],test_predictions[tend:-1,0],testlabel[tend:-1,0],wwee)
                rat1=losstrend10-losstrend100*wwee
                rat2=losstrend20-losstrend200*wwee
                rat11=losstrend100/losstrend10
                rat22=losstrend200/losstrend20
                losseset=losstrend00 /5+(losstrend100*wwee*2+losstrend200*wwee*2)/5

                [losstrend,losstrend1]=pp.losstrendv2(testout_data[ttsize:test_data.shape[0],:],test_predictions[ttsize:test_data.shape[0],:],testlabel[ttsize:test_data.shape[0],:],wwee)
                [losstrendtr,losstrendtr1]=pp.losstrendv2(trainout_data[trsize:train_data.shape[0],:],train_predictions[trsize:train_data.shape[0],:],trainlabel[trsize:train_data.shape[0],:],wwee)

            else:
                losstrend=0
                losstrend1=0
                losstrendold=0
                losstrendold1=0
            checkk=1

            if len(l1rall)==1 or okk==1:
                if traininit==0:
#                    test_data_norminall=test_data_normin
                    testlabelall=testlabel
                    if np.min((testdataall[:,selectrainout[0]]))>minn:
                        testout_dataall=(testdataall[:,selectrainout[0]])-minn
                else:
                    if np.min((testdataall[:,selectrainout[0]]))>minn:
                        testout_dataall=(testdataall[:,selectrainout[0]])-minn
                    tend=int(ttsize+(testdataall.shape[0]-ttsize)/3*1.5)
                    tend2=int(ttsize+(testdataall.shape[0]-ttsize)/3*3)
                print(str(np.min(testout_dataall)))
                loaded_model=model
                test_predictions0 = loaded_model.predict([test_data_normin,testlabel])
                errormape=np.mean(abs(test_predictions0[0:ttsize-1].flatten()-(testout_data[0:ttsize-1].flatten()))/(testout_data[0:ttsize-1].flatten()))*100
                print("Testing MAPE: {:10.5f}".format(errormape))
                plt.plot(testout_data[0:ttsize-1],test_predictions0[0:ttsize-1],'o')
                test_predictions = loaded_model.predict([test_data_norminall,testlabelall])                
                findrow=np.argwhere(testdataall[0:ttsize-1,34]>0).T[0]
                errormapes=np.mean(np.abs(testout_dataall[findrow].flatten()-test_predictions[findrow].flatten())/testout_dataall[findrow].flatten())
#                plt.plot(preds, testdata[findrow,selectrainout],'*')
                print("SHEARED FLOW Testing set Mean Abs Error: {:10.5f}".format(errormapes))
                findrow=np.argwhere(testdataall[0:ttsize-1,34]==0).T[0]
#                loss = model.evaluate([test_data_normin[findrow,:],testlabel[findrow]], testdata[findrow,selectrainout], verbose=0)
                errormapeu=np.mean(np.abs(testout_dataall[findrow].flatten()-test_predictions[findrow].flatten())/testout_dataall[findrow].flatten())
                print("UNIFORM FLOW Testing set Mean Abs Error: {:10.5f}".format(errormapeu))
                
                bdivednew=[]
                infnew=[]
                for ii in range(0,4):
                    if ii==0:
                        bu=np.argwhere(testdata[:,40]==1)  
                    elif ii==1:
                        bu=np.argwhere(testdata[:,40]<1)     
                        
                    elif ii==2:
                        bu=np.argwhere((testdata[:,4]>9)) 
                    elif ii==3:
                        bu=np.argwhere((testdata[:,4]>9) &(testdata[:,34]>0)) 
                    if len(bu)>0:
                        buc=bu[(bu<tend) & (bu>ttsize)]
                        bur=bu[bu>tend]
                        difff=np.abs(np.sign(np.abs(testlabel)-thres)).flatten()
                        cee=np.sum(np.sign(test_predictions0[buc].flatten()-(testout_data[buc].flatten()))+1)/2/len(buc)
                        ree=1-np.sum(np.sign(test_predictions0[bur].flatten()-(testout_data[bur].flatten()))+1)/2/len(bur)
#                        cee1=np.sum((1-difff[buc])*(np.sign(test_predictions[buc].flatten()-(testout_data[buc].flatten()))+1))/2/len(buc)
#                        ree1=np.sum((1-difff[bur])*(np.sign(-test_predictions[bur].flatten()+(testout_data[bur].flatten()))+1))/2/len(bur)
#                        cee2=np.sum((difff[buc])*(np.sign(-test_predictions[buc].flatten()+(testout_data[buc].flatten()))+1))/2/len(buc)
#                        ree2=np.sum((difff[bur])*(np.sign(test_predictions[bur].flatten()-(testout_data[bur].flatten()))+1))/2/len(bur)
#                        cee=cee1+cee2
#                        ree=ree1+ree2

                        if ii<=1:
                            bdivednew=bdivednew+[len(buc)/(tend-ttsize),cee,ree]
                        elif ii>=2:
                            bue=bu[(bu<ttsize)]
                            errormapenew0=np.mean(abs(test_predictions0[bue]-(testout_data[bue]))/(testout_data[bue]))*100
                            infnew=infnew+[errormapenew0,cee,ree]
                    elif len(bu)==0 and uuniform<=5:
                        bdivednew=bdivednew+[0,0,0]                

                mapestat=0
                mapenon=0
                mapestat2=0
                mapenon2=0
                mapeaa= model.evaluate([test_data_normin,testlabel], testdata[:,selectrainout[0]])
                if traininit==1:
                    mapeaa=mapeaa[0]
                    mape=mape
                if traininit==0:
                    out=[curlearn,entp1,entp2,entp3,mapestat,mapenon,mapestat2, mapenon2, mapeaa,ttsize,wwee,minn,losstrend,errormape,losstrendtr,errormapetr,losstrendold,losstrendold1,losstrend1,errormapeold,errormapesold,errormapeuold,errormapes,errormapeu]
                    out=out+bdivedold+bdivednew+infold+infnew
                else:
                    out=[curlearn,entp1,entp2,entp3,mapestat,mapenon,mapestat2, mapenon2, mapeaa,ttsize,wwee,minn,losstrend,errormape,mape,mapetrain,errormapes,errormapeu]
                    out=out+bdivednew
                print('out',str(out))
#        out=[minn,losstrend,errormape,mape,mapetrain]
    
#                print('outconcerRned',str(out))
            if checkk==1 and (len(l1rall)==1 or okk==1) and (uuniform>=1 and uuniform<=5):
                test_predictions = model.predict([test_data_normin,testlabel])                
                ## UNIFORM VS SHEARED
                if len(np.argwhere(testdata[0:ttsize-1,34]>0))>0:
                    findrow=np.argwhere(testdata[0:ttsize-1,34]>0).flatten()
    #                loss= model.evaluate([test_data_normin[findrow,:],testlabel[findrow]], testdata[findrow,selectrainout], verbose=0)
                    errormapes=np.mean(np.abs(testout_data[findrow]-test_predictions[findrow])/testout_data[findrow])
                    preds= model.predict([test_data_normin[findrow,:],testlabel[findrow]])
                else:
                    errormapes=0
#                plt.plot(preds, testdata[findrow,selectrainout],'*')
                print("SHEARED FLOW Testing set Mean Abs Error: {:10.5f}".format(errormapes))
                if len(np.argwhere(testdata[0:ttsize-1,34]==0))>0:

                    findrow=np.argwhere(testdata[0:ttsize-1,34]==0).flatten()
    #                loss = model.evaluate([test_data_normin[findrow,:],testlabel[findrow]], testdata[findrow,selectrainout], verbose=0)
                    errormapeu=np.mean(np.abs(testout_data[findrow]-test_predictions[findrow])/testout_data[findrow])
                    predu= model.predict([test_data_normin[findrow,:],testlabel[findrow]])
                else:
                    errormapeu=0
                print("UNIFORM FLOW Testing set Mean Abs Error: {:10.5f}".format(errormapeu))
                
                ainput=test_data
                binput=test_data_normin
                cinput=testdata
                dinput=testlabel
                xlinear1=np.linspace(min(testout_data[0:ttsize-1,0]+minn)-correct,max(testout_data[0:ttsize-1,0]+minn)-correct,30)
                ylinear1=xlinear1
                fig = plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
                ax = fig.add_subplot(111)
                findrowu=np.argwhere(cinput[0:ttsize-1,34]==0).T[0]
                loss = model.evaluate([binput[findrowu,:],dinput[findrowu]], cinput[findrowu,selectrainout], verbose=0)
                metric=loss
                predu= model.predict([binput[findrowu,:],dinput[findrowu]])
                errors=abs(cinput[findrowu,selectrainout]-predu.T)[0]
                eee=np.argsort(errors)
                prp=1
                print('newerr',str(np.mean(errors[eee[0:int(prp*len(eee))]])))
                plt.plot(cinput[findrowu[eee[0:int(prp*len(eee))]],selectrainout]+minn,predu[eee[0:int(prp*len(eee))],:]+minn,'b+', markersize=12)
#                print("Uniform Testing set Mean Abs Error: {:10.5f}".format(metric))
                findrows=np.argwhere(cinput[0:ttsize-1,34]>0).T[0]
                loss= model.evaluate([binput[findrows,:],dinput[findrows]],cinput[findrows,selectrainout], verbose=0)
                metric=loss
#                print("Sheared Testing set Mean Abs Error: {:10.5f}".format(metric))
                preds= model.predict([binput[findrows,:],dinput[findrows]])
                errors=abs(cinput[findrows,selectrainout]-preds.T)[0]
                eee=np.argsort(errors)
                prp=.97
                plt.plot(cinput[findrows[eee[0:int(prp*len(eee))]],selectrainout]+minn,preds[eee[0:int(prp*len(eee))],:]+minn,'go', fillstyle='none', markersize=10)
                plt.plot(xlinear1,ylinear1, 'r')
                ax.legend(['Uniform flow','Sheared flow'], fontsize = 20)
                plt.xlabel('Measured output', fontsize = 20)
                plt.ylabel('Predicted output', fontsize = 20)
                plt.xticks(fontsize=20)  
                plt.yticks(fontsize=20)
                plt.grid()
                if importwave==0:
                    stringsave='./sutest'+str(nnnum)+str(testnumprint)+'full'+str(full)+'out'+str(selectrainout[0])
                else:
                    stringsave='./shellsutest'+str(nnnum)+str(testnumprint)+'full'+str(full)+'wave'+str(importwave)+'out'+str(selectrainout[0])+'l1r'+str(l1rall[l1sel])+'vscale'+str(vscalee)+'curlearn'+str(curlearn)+str(ttsize)+'u'+str(uuniform)
                if mapee>=2:
                    stringsave=stringsave+'mape'+str(mapee)+'ww'+str(wweet)
                if addadd>0:
                    stringsave=stringsave+str(selother)
                plt.savefig(stringsave+'.png', dpi=350) 
               
                np.corrcoef(cinput[findrowu,14],cinput[findrowu,15])
                np.corrcoef(cinput[findrows,14],cinput[findrows,15])
               
                ''' end test shap lime'''
                ## SEPARATE TO 30 AND 80 MM PIPES
#                ainput=traindata[0:int(ttsize*splitt0/(1-splitt0)) ,:]
#                binput=train_data_normin[0:int(ttsize*splitt0/(1-splitt0)) ,:]
#                findrow=np.argwhere((ainput[:,13] <=np.max(testnum30u)) & (ainput[:,13] >=np.min(testnum30u))).T[0]
#                [loss30u, metric30u] = model.evaluate(binput[findrow,:],ainput[findrow,selectrainout], verbose=0)
#                findrow=np.argwhere((ainput[:,13] <=np.max(testnum30s)) &(ainput[:,13] >=np.min(testnum30s))).T[0]
#                [loss30s, metric30s] = model.evaluate(binput[findrow,:], ainput[findrow,selectrainout], verbose=0)
#                findrow=np.argwhere((ainput[:,13] <=np.max(testnum40u)) & (ainput[:,13] >=np.min(testnum40u))).T[0]
#                [loss40u, metric40u] = model.evaluate(binput[findrow,:], ainput[findrow,selectrainout], verbose=0)
#                findrow=np.argwhere((ainput[:,13] <=np.max(testnum40s)) &( ainput[:,13] >=np.min(testnum40s))).T[0]
#                [loss40s, metric40s] = model.evaluate(binput[findrow,:], ainput[findrow,selectrainout], verbose=0)
#                metric50u=0
#                findrow=np.argwhere((ainput[:,13] <=np.max(testnum50u)) &( ainput[:,13] >=np.min(testnum50u))).T[0]
#                if len(findrow)>0:
#                    [loss50u, metric50u] = model.evaluate(binput[findrow,:], ainput[findrow,selectrainout], verbose=0)
#                print('Train losses',str([metric30u,metric30s,metric40u,metric40s,metric50u]))
#                
#                ainput=testdata[0:ttsize,:]
#                binput=test_data_normin[0:ttsize,:]
#                
#                findrow=np.argwhere((ainput[:,13] <=np.max(testnum30u)) & (ainput[:,13] >=np.min(testnum30u))).T[0]
#                [loss30u, metric30u] = model.evaluate(binput[findrow,:],ainput[findrow,selectrainout], verbose=0)
#                findrow=np.argwhere((ainput[:,13] <=np.max(testnum30s)) &(ainput[:,13] >=np.min(testnum30s))).T[0]
#                [loss30s, metric30s] = model.evaluate(binput[findrow,:], ainput[findrow,selectrainout], verbose=0)
#                findrow=np.argwhere((ainput[:,13] <=np.max(testnum40u)) & (ainput[:,13] >=np.min(testnum40u))).T[0]
#                [loss40u, metric40u] = model.evaluate(binput[findrow,:], ainput[findrow,selectrainout], verbose=0)
#                findrow=np.argwhere((ainput[:,13] <=np.max(testnum40s)) &( ainput[:,13] >=np.min(testnum40s))).T[0]
#                [loss40s, metric40s] = model.evaluate(binput[findrow,:], ainput[findrow,selectrainout], verbose=0)
##                findrow=np.argwhere((ainput[:,13] <=np.max(testnum50u)) &( ainput[:,13] >=np.min(testnum50u))).T[0]
#                [loss50u, metric50u] = model.evaluate(binput[findrow,:], ainput[findrow,selectrainout], verbose=0)
#                print('Test losses',str([metric30u,metric30s,metric40u,metric40s,metric50u]))
                ## 
            """start output weights"""     
            
            # get the first layer weight
            count=1
            for layer in model.layers:
                if count==1:
                    weights = layer.get_weights() # list of numpy arrays
                if count==2:
                    weights2 = layer.get_weights() # list of numpy arrays  
                if count==3:
                    weights3 = layer.get_weights() # list of numpy arrays          
                if count==4:
                    weights4 = layer.get_weights() # list of numpy arrays        
                if count==5:
                    weights5 = layer.get_weights() # list of numpy arrays    
                if count==6:
                    weights6 = layer.get_weights() # list of numpy arrays          
                count=count+1
            encoder_weights=weights2[0]
            encoder_bias=weights2[1]
            count=0
            l=model.layers[1]
            neuronw=np.sqrt(np.sum(np.abs(l.get_weights()[0]**2), axis=1))
            nsort=np.argsort(neuronw)## INCREASING ORDER
            #plt.plot(neuronw[nsort])
            nsort=nsort.astype(int).T
            if iterr==0:
                neuronw0=neuronw+0
            """plot figure"""       
            if iterr==0 or pruning==0:
                fig = plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
                ax = fig.add_subplot(111)
                plt.bar(range(1,1+len(neuronw)), neuronw, align='center', alpha=0.5)
                #plt.xticks(y_pos, objects)
                plt.ylabel('Weights', fontsize = 16)
                plt.xlabel('Variable index', fontsize = 16)
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))   
                plt.xticks(fontsize=16)  
                plt.yticks(fontsize=16)
                plt.grid()
                plt.rc('font',family='Times New Roman')        
            #    ax.legend(['Test data'], fontsize = 20)
                plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)    
                if vscalee==3:
                    thres0=.04
                else:
                    thres0=.04
                if selectrainout[0]==0:
                    nnss=1
                    thres0=.08
                elif selectrainout[0]==0 and paramstotune[5]==30:
                    nnss=3
                    thres0=.08
                elif selectrainout[0]==1:
                    nnss=3
                    thres0=.04
                else:
                    nnss=10
                if uuniform==3:
                    nnss=1
                    thres0=.05
#                nnss=nscale
#                neuronww0=neuronww+0
                neuronww=neuronw+0  #+.1
                if len(l1rall)>1:        
                    for nnww in range(0,len(selecttall)):
                        if selecttall[nnww] in rscale:
                            neuronww[nnww]=0#neuronw[nnww]*nnss
                        findrow=np.argwhere(neuronww**2>=np.sum(np.sort(neuronww**2))*0)
                else:
                    findrow=np.argwhere(neuronww**2>=0)#np.sum(np.sort(neuronww**2))*thres/100000000000000)
                findrow=findrow.T
                findrow=findrow[0]
                numsens=findrow.shape[0]
                for ff in range(0,numsens):
                    if ff==0:
                        selecttnew=int(selectt[findrow[ff]])
                        selecttnew=np.hstack((selecttnew,))
                    else:
                        selecttnewt=int(selectt[findrow[ff]])
                        selecttnew=np.hstack((selecttnew,selecttnewt))
                selecttnew=list(selecttnew)
                if len(l1rall)>1:
                    selecttnew=list(np.unique(np.asarray(list(selecttnew)+rscale)))
#            if iterr>=0:            
#                f = K.function([model.layers[0].input, K.learning_phase()],
#                               [model.layers[-1].output])
#                [pred,uncert]=predict_with_uncertainty(f, [test_data_normin,testlabel], 1,1000)  ## dimension of output, number of iterations
#                testpredl=pred-uncert  
#                testpredu=pred+uncert
#        
#                varr=uncert**2
#                l=1
#                NN=test_data_normin.shape[0]
#                test_predictions=test_predictions.astype(np.float32).reshape((-1,1))
#                train_predictions=train_predictions.astype(np.float32).reshape((-1,1))
##                inputt_predictions=inputt_predictions.astype(np.float32).reshape((-1,1))
#                pred=pred.astype(np.float32).reshape((-1,1))
#                """ COMPARISON BETWEEN PREDICTION AND TRUE OUTPUT"""
##                [predtr,uncerttr]=predict_with_uncertainty(f, train_data_normin, 1,1000)  ## dimension of output, number of iterations
##                cffind=[index for index,value in enumerate(testdata[:,14]) if value > .9]
##                ilfind=[index for index,value in enumerate(testdata[:,15]) if value > .9]
##                infind=list(set(cffind).intersection(ilfind))
##                infind=np.asarray(infind)
##                num=0
##                metrictr=np.mean(abs(predtr.T-trainout_data[:,num])/trainout_data[:,num])*100
##                if countitert==0:
##                    adjust=0
##                else:
##                    adjust=minn
##                metric2=np.mean(abs(test_predictions.T-(testout_data[:,num]-adjust))/(testout_data[:,num]-adjust))*100
##                metric2=np.mean(abs(pred.T-(testout_data[:,num]-adjust))/(testout_data[:,num]-adjust))*100
##                metric3=np.mean(abs(pred[infind].T-(testout_data[infind,num]-adjust))/(testout_data[infind,num]-adjust))*100
##                print("Testing set Mean Abs Error: {:7.2f}".format(metric2))
##                print("Testing set Mean Abs Error remove nonstat: {:7.2f}".format(metric3))
#            print('min=',str(minn))
            
#            if iterr==np.max(xtall) and pruning>=0:    
#                
#                """ AA"""
#                for num in range(0,testout_data.shape[1]):
#                    if countitert==0:
#                        testout_data[:,num]=testout_data[:,num]
#                    test_predictions[:,num]=test_predictions[:,num]
##                    pred=pred
#                    
#                    xlinear1=np.linspace(min(trainout_data[0:int(ttsize*splitt0/(1-splitt0))-1,num]+minn)-correct,max(trainout_data[0:int(ttsize*splitt0/(1-splitt0))-1,num]+minn)-correct,30)
#                    ylinear1=xlinear1
#                    fig = plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
#                    ax = fig.add_subplot(111)
#                    plt.xlabel('Measured output', fontsize = 20)
#                    plt.ylabel('Predicted output', fontsize = 20)
#                    prp=.97
#                    preds= model.predict([train_data_normin,trainlabel])
#                    errors=abs(trainout_data.T-preds.T)/trainout_data.T
#                    eee=np.argsort(errors[0,:])                    
#                    plt.plot(trainout_data[0:int(ttsize*splitt0/(1-splitt0))-1,selectrainout]+minn,preds[0:int(ttsize*splitt0/(1-splitt0))-1,:]+minn,'b*',xlinear1,ylinear1, 'r')
#
##                    plt.plot(trainout_data[eee[0:int(prp*len(eee))],selectrainout]+minn,preds[eee[0:int(prp*len(eee))],:]+minn,'b*',xlinear1,ylinear1, 'r')
##                    plt.plot(trainout_data[:,num]+minn-correct,train_predictions[:,num]+minn-correct,'b*',xlinear1,ylinear1, 'r')
#                    plt.xticks(fontsize=20)  
#                    plt.yticks(fontsize=20)
#                    plt.grid()
#                    plt.rc('font',family='Times New Roman')        
##                    ax.legend(['Train data'], fontsize = 20)
#
#                    if importwave==0:
#                        stringsave='./train'+str(nnnum)+str(testnumprint)+'full'+str(full)+'out'+str(selectrainout[0])
#                    else:
#                        stringsave='./shelltrain'+str(nnnum)+str(testnumprint)+'full'+str(full)+'wave'+str(importwave)+'out'+str(selectrainout[0])+'l1r'+str(l1rall[l1sel])+'vscale'+str(vscalee)+'curlearn'+str(curlearn)+str(ttsize)
#                    if mapee>=2:
#                        stringsave=stringsave+'mape'+str(mapee)+'ww'+str(wwee)
#                    plt.savefig(stringsave+'.png', dpi=350) 
#
#                    xlinear1=np.linspace(min(testout_data[:,num]+minn)-correct,max(testout_data[:,num]+minn)-correct,30)
#                    ylinear1=xlinear1
#                    fig = plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
#                    ax = fig.add_subplot(111)
#                    plt.xlabel('Measured output', fontsize = 16)
#                    plt.ylabel('Predicted output', fontsize = 16)
#                    prp=.97
#                    
#                    preds= model.predict([test_data_normin[0:ttsize-1,:],0*testlabel[0:ttsize-1,:]])
#                    errors=abs(testout_data[0:ttsize-1].T-preds.T)/testout_data[0:ttsize-1].T
#                    eee=np.argsort(errors[0,:])                    
#                    plt.plot(testout_data[0:ttsize-1,selectrainout]+minn,preds[0:ttsize-1,:]+minn,'b*',xlinear1,ylinear1, 'r')
#                    plt.xticks(fontsize=16)  
#                    plt.yticks(fontsize=16)
#                    plt.grid()
#                    plt.rc('font',family='Times New Roman')        
#                    ax.legend(['Test data'], fontsize = 20)
#                    if importwave==0:
#                        stringsave='./test'+str(nnnum)+str(testnumprint)+'full'+str(full)+'out'+str(selectrainout[0])
#                    else:
#                        stringsave='./shelltest'+str(nnnum)+str(testnumprint)+'full'+str(full)+'wave'+str(importwave)+'out'+str(selectrainout[0])+'l1r'+str(l1rall[l1sel])+'vscale'+str(vscalee)+'curlearn'+str(curlearn)+str(ttsize)+'u'+str(uuniform)
#                    if mapee>=2:
#                        stringsave=stringsave+'mape'+str(mapee)+'ww'+str(wwee)
##                plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)    
#                plt.savefig(stringsave+'.png', dpi=350) 
#    
                l1r=l1rall[l1sel]
                num=0
                nnn=range(0,testout_data.shape[0])
                fig = plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
                ax = fig.add_subplot(111)
                plt.plot(testout_data[nnn,num]+minn)
                plt.plot(test_predictions[nnn,num]+minn)

                mapetrainout=[mapetrainout]
                mape=[mape]
                adjustt=0
#                errorvar=np.mean(abs(inputt_predictions.T-(inputt[:,selectrainout[0]]-adjustt))**2)
                ## FOR TEST DATA CHECK!
#                errormape=(abs(inputt_predictions.T-(inputt[:,selectrainout[0]]-adjustt))/(inputt[:,selectrainout[0]]-adjustt))*100
    #            errormape=(abs(pred.T-(testout_data[:,num]-adjust))/(testout_data[:,num]-adjust))*100
#                errormape_mean=np.mean(errormape)
#                errormape_std=np.std(errormape)
#                errormape_pct1=np.percentile(errormape,50)
#                errormape_pct2=np.percentile(errormape,75)
#                print('TEST ERROR MEAN AND STD, PCT',str([errormape_mean,errormape_std,errormape_pct1,errormape_pct2]))

#                for num in range(0,testout_data.shape[1]):
#                    fig = plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
#                    ax = fig.add_subplot(111)
#                #    ax.plot(testout_data[:,num],pred,'*',xlinear1,ylinear1, lw=2,  color='blue')
#                    ax.plot(xlinear1,ylinear1, 'r')    
#                    plt.errorbar(testout_data[:,num]+minn-correct,test_predictions[:,num]+minn-correct, yerr=uncert, fmt='o')
#                    plt.xticks(fontsize=16)  
#                    plt.yticks(fontsize=16)       
#                    plt.rc('font',family='Times New Roman')        
#                #    ax.fill_between(testout_data[:,num],testpredl[:,num],testpredu[:,num],color='black',alpha=0.6) 
#                    plt.xlabel('Measured output', fontsize = 16)
#                    plt.ylabel('Predicted output', fontsize = 16)
#                    plt.grid()      
#                    np.mean((testout_data[:,num]-test_predictions[:,num])**2)
#        #            plt.xlim([0.2,0.8])
#        #            plt.ylim([0.2,0.8])
#                    ax.legend(['Test data'], fontsize = 20)
#                    if importwave==0:
#                        stringsaveu='./testuncert'+str(nnnum)+str(testnumprint)+'full'+str(full)+'out'+str(selectrainout[0])
#                    else:
#                        stringsaveu='./shelltestuncert'+str(nnnum)+str(testnumprint)+'full'+str(full)+'wave'+str(importwave)+'out'+str(selectrainout[0])+'l1r'+str(l1rall[l1sel])+'vscale'+str(vscalee)+'curlearn'+str(curlearn)+str(ttsize)+'u'+str(uuniform)
#                    if mapee>=2:
#                        stringsaveu=stringsave+'mape'+str(mapee)
#                plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)    
#                plt.savefig(stringsaveu+'.png', dpi=350) 
        ## CHECK MOST PROBLEMATIC DATAPOINT
                findrow=np.argwhere(neuronw>0)
                findrow=findrow.T
                numsens=findrow[0].shape[0]
                print('numsens'+str(numsens))
                if importwave==2:
                    stringsavepenal='./currentanal'+str(nnnum)+str(testnumprint)+'full'+str(full)+'out'+str(selectrainout[0])+'l1r'+str(l1r)+'vscale'+str(vscalee)+'curlearn'+str(curlearn)+str(ttsize)+'u'+str(uuniform)+'mape'
                else:
                    stringsavepenal='./currentanal'+str(importwave)+str(nnnum)+str(testnumprint)+'full'+str(full)+'out'+str(selectrainout[0])+'l1r'+str(l1r)+'vscale'+str(vscalee)+'curlearn'+str(curlearn)+str(ttsize)+'u'+str(uuniform)+'mape'
    
                if mapee>=2:
                    stringsavepenal=stringsavepenal+'mape'+str(mapee)+'ww'+str(wweet)
                if addadd==1:
                    stringsavepenal=stringsavepenal+str(selother)
                if traininit==0:
                    df=pd.concat([pd.DataFrame({numsens}),  pd.DataFrame({mape[0]}),pd.DataFrame({'weights':neuronw}),pd.DataFrame({loss}),pd.DataFrame({'sensnum':selectt}),pd.DataFrame({'weights':neuronw0}),pd.DataFrame({'outs':out})], axis=1)
                else:
                    df=pd.concat([pd.DataFrame({numsens}),  pd.DataFrame({mape[0]}),pd.DataFrame({'weights':neuronw}),pd.DataFrame({loss[0]}),pd.DataFrame({'sensnum':selectt}),pd.DataFrame({'weights':neuronw0}),pd.DataFrame({'outs':out})], axis=1)
                if len(selectt)<2:
                    df.to_csv(stringsavepenal+'.csv',index=False,sep=',')        
                else:
                    df.to_csv(stringsavepenal+'2.csv',index=False,sep=',')        
                
#            errormape=(abs(inputt_predictions.T-(inputt[:,num]-adjust))/(inputt[:,num]-adjust))*100
##            errormape=(abs(pred.T-(testout_data[:,num]-adjust))/(testout_data[:,num]-adjust))*100
#            errormape_mean=np.mean(errormape)
#            
#            
#            nsort=np.argsort(errormape).flatten()
#            nsort=list(nsort)
#            
#            errormape2=np.zeros((1,len(nsort)))
#            errortest=np.zeros((len(nsort),inputt.shape[1]))
#            
#            errormape2=errormape[0,nsort]
##            errortest=test_data_full[nsort,:]
#            errormeas=inputt[nsort,:] ##  sorted input
#            errorpred=inputt_predictions[nsort]
#            testnumcheck=errormeas[-1,13]
#            plt.plot(errormeas[:,13],errormape2,'o')
##            sio.savemat('error'+str(testnumprint)+str(selectrainout[0])+'.mat', {'errormeas':errormeas})
#                        
##            testnumcheck=4114
#            print('check test No.',  testnumcheck)
#            findrow=list(np.argwhere((errormeas[:,13]==testnumcheck)).flatten())
#            findrow=np.sort(np.unique(findrow))
            
            findrow=list(np.argwhere((inputt[:,13]==3002)).flatten())
            findrow=np.sort(np.unique(findrow))
#            score =model.evaluate(inputt_data_normin,inputt[:,selectrainout[0]], verbose=0)
#            print("%s: %.2f%%" % (model.metrics_names[1], score[1]))
            

            findrow=np.argwhere(testdatastat[:,15]<.85)
            aa=testdatastat[findrow,13].flatten()
            """ ERROR SHADES"""
            if importshear==1:
    #            testnumtolabelall=[3004,3005,3007,3008,3009,3010,3012,3016,3017,3106,3107]  ## shell
                testnumtolabelall=[101,102,103,104,105,106,107]## SJTU sheared
                testnumtolabel=testnumtolabelall[0]
                data= sio.loadmat('dataall'+str(testnumtolabel)+'.mat')  
                train=sio.loadmat('traindata'+str(testnumtolabel)+'.mat')  
                test=sio.loadmat('testdata'+str(testnumtolabel)+'.mat') 
                inputttolabel=data['dataall']
                traindatatolabel=train['traindata']
                testdatatolabel=test['testdata']
                Numstart=311
                Numstart1=1
                inputttolabel=inputttolabel[len(inputttolabel)-Numstart:len(inputttolabel)]
                Numall=311
    #            inputttolabel=inputttolabel[5800:len(inputttolabel)]
                
                
                inputttolabel_normin=(inputttolabel[:,selectt]-meanstats[selectt,0])/meanstats[selectt,1]            
                inputttolabel_predictions=model.predict(inputttolabel_normin)   
                xlinear1=np.linspace(min(trainout_data[:,num]),max(trainout_data[:,num]),30)
                ylinear1=xlinear1
                fig = plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
                ax = fig.add_subplot(111)
                plt.xlabel('Measured output', fontsize = 16)
                plt.ylabel('Predicted output', fontsize = 16)
                plt.plot(np.sqrt(1)*inputttolabel[:,num],inputttolabel_predictions,'b*',xlinear1,ylinear1,  'r')
                plt.grid()      
                ax.legend(['Shell test'], fontsize = 16)
                
                [pred,uncert]=predict_with_uncertainty(f, inputttolabel_normin, 1,1000)  ## dimension of output, number of iterations
                testpredl=pred-uncert  
                testpredu=pred+uncert
                fig = plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
                ax = fig.add_subplot(111)
            #    ax.plot(testout_data[:,num],pred,'*',xlinear1,ylinear1, lw=2,  color='blue')
            #    ax.legend(loc='upper left')
            #    ax.fill_between(testout_data[:,num],testpredl[:,num],testpredu[:,num],alpha=0.35)  
                plotx=[(x+Numstart1)/Numall for x in range(len(pred))]
                ax.plot(plotx,inputttolabel[:,num],'r*',plotx,pred,'bs',plotx,inputttolabel[:,1],'gs',plotx,inputttolabel[:,2],'ko')
                ax.fill_between(plotx,testpredl[:,num],testpredu[:,num],alpha=0.5)    
            #    ax.plot(test_data_normin[:,1],pred,'*')
            #    ax.legend(loc='upper left')
            #    ax.fill_between(test_data_normin[:,1],testpredl[:,num],testpredu[:,num],alpha=0.35)    
                
                plt.xlabel('z/L', fontsize = 16)
                plt.ylabel('A/D', fontsize = 16)
                plt.grid()                
                ax.legend(['Measurement CF 1 time','Prediction mean CF dominant','Measurement IL dominant','Measurement CF 3 times'], fontsize = 16)
                stringsave='./compare'+str(nnnum)+str(testnumprint)+'full'+str(full)+str(testnumtolabel)+'out'+str(selectrainout[0])+'vscale'+str(vscale)+'curlearn'+str(curlearn)+str(ttsize)+'u'+str(uuniform)+'mape'+str(mapee)+'ww'+str(wweet)
                plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)    
                plt.savefig(stringsave+'.png', dpi=350) 
    ## SAVE NEURAL NETWORK MODEL
            if savemodel==1:
                # serialize model to JSON
                model_json = model.to_json()
                with open(stringsavemod+'.json', "w") as json_file:
                    json_file.write(model_json)
                # serialize weights to HDF5
                model.save_weights(stringsavemod+'.h5')
                print("Saved model to disk")
                 
                # later...
                 
                # load json and create model
                json_file = open(stringsavemod+'.json', 'r')
                
                loaded_model_json = json_file.read()
                json_file.close()
                if mapee<2:
                    loaded_model = tf.keras.models.model_from_json(loaded_model_json,custom_objects={'L21': L21(l1r)})
                else:
                    
                
                    sio.savemat(stringsavemod+'.mat', {'inp':[noutput,nnnum,paramstotune,l1r,pruning,train_data,mapee,thres]})
#                    inp=sio.loadmat(stringsave+'.mat')
#                    inp=inp['inp'][0]
#                    [model,droprat,regurat,layernums,l1r] = pp.build_modelmapee(inp[0][0][0],inp[1][0][0],inp[2][0],inp[3][0][0],inp[4][0][0],inp[5],inp[6][0][0],inp[7][0][0])
#                    model.summary()   
                    model.save_weights(stringsavemod+'.h5')
                    loaded_model=model
                    score = loaded_model.evaluate([test_data_normin[0:ttsize-1,:],testlabel[0:ttsize-1,0]], testout_data[0:ttsize-1], verbose=0)
                    scoreT = loaded_model.evaluate([test_data_normin[ttsize:-1,:],testlabel[ttsize:-1]], testout_data[ttsize:-1], verbose=0)
 
                    if traininit==1:
                        print("%s: %.2f%%" % (loaded_model.metrics_names, score[0]))
                        print("%s: %.2f%%" % (loaded_model.metrics_names, scoreT[0]))
                        
                    else:
                        print("%s: %.2f%%" % (loaded_model.metrics_names, score))
                        print("%s: %.2f%%" % (loaded_model.metrics_names, scoreT))
                
#                test_predictions0 = model.predict([test_data_normin[0:ttsize-1,:],testlabel[0:ttsize-1,0]])
#                plt.plot(test_predictions0,testout_data[0:ttsize-1],'o')
#                test_predictions1 = model.predict([test_data_normin[ttsize:-1,:],testlabel[ttsize:-1,0]])
#                plt.plot(test_predictions1,testout_data[ttsize:-1],'+')
#                rf = RandomForestRegressor(n_estimators=50, random_state=20)
#                rf.fit(test_data_normin[0:ttsize-1,:], test_predictions0 )
#                print(rf.score(test_data_normin[0:ttsize-1,:], test_predictions0))
#                findrow=np.random.choice(ttsize-1,ttsize-1,replace='false')   
#                [coeff,interpp,scoreaa,input_descrp_totab0]=pp.explainn(test_data[findrow,:],test_data_normin[findrow,:],selectt,selectrainout,rf,inputt_descrp)  
#                findrow0=np.argwhere(scoreaa>=0.5)[:,0]
#        #        findrow1=intersection(findrow0,np.argwhere(test_data[:,sel]<0.001)[:,0])
#                if len(np.argwhere(np.asarray(selectt)==34))>0:
#                    sel=np.argwhere(np.asarray(selectt)==34)[0][0]
#                    findrow1=intersection(findrow0,np.argwhere(test_data[:,sel]>0)[:,0])
#                    intersection(findrow1,np.argwhere(testdata[:,15]>0.7)[:,0])
#                    intersection(findrow1,np.argwhere(testdata[:,14]>0.7)[:,0])
#                    findrow=findrow1
#                findrow=findrow0
#                input_descrp_out=inputt_descrp[selectrainout[0]]
#        #        selecttall=common_member([1,14,6,8,34],selectt)
#                selecttall=selectt
#                pp.explainnplot(test_data,selecttall,selectrainout,model,input_descrp_totab0,coeff,findrow,testout_data+minn,input_descrp_out)  
#
            savemeanstats=0
            if savemeanstats==1:
                mime=np.vstack((meanstats,np.asarray([minn,minn])))
                sio.savemat(stringsave+'.mat', {'mime':mime})
                
            """ CONTROL VARIABLES ON UNLABELED DATA"""
            if full!=1 and backward==0 and plotdetail==1:      
                num=0  ## THIS CAN BE ADJUSTED WHEN OUTPUT IS MORE THAN 1
                nn=100
                ref=[0.6,0.1,0.1,0.3,0.1,0.1,.1,.2,40000,6.6, 3.3, 90,90, 90, 0.8,0.8,3000,0,0.8,0.8,.5,1,0,1,0,0,0,1,0,0,0,1,0,1,0.0001,0.85,0.85,100,1300,0.01,0]  
    #            if 0 in selectt:
    #                ref[0]=float("{:.2f}".format(np.mean(inputt[:,0])))
    #            
    #            if 1 in selectt:
    #                ref[1]=float("{:.2f}".format(np.mean(inputt[:,1])))
    #            if 10 in selectt:
    #                ref[10]=float("{:.2f}".format(np.mean(inputt[:,10])))
    #            linerats=[.88,1,1.17]
                
                linerats=[.6,1,1.25]
                linen=len(linerats)
                for nppvar in range(1,len(selectt)): ## VARY nppvar to adjust the variable that varies with the line
        #            for npp in range(0,len(selectt)):
                    for npp in range(0,1):
                        xx0=np.zeros((nn,len(selectt)))
                        sel=selectt[npp]
                        x3=np.linspace(min(inputt[:,sel])-0.00000000001,max(inputt[:,sel])+0.000000000001,nn) ## A RANGE OF VALUES
                        xx0[:,npp]=(x3-meanstats[sel,0])/meanstats[sel,1]
        
                        fig = plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')             
                        ax = fig.add_subplot(111) 
                        for lnn in range(0,linen):
                            strings0=''
                            for ii in range(0,len(selectt)):
                                if ii!=npp:
                                    if ii==nppvar:
                                        ratt=linerats[lnn]
                                    else:
                                        ratt=1
                                    xtemp=(ref[selectt[ii]]*ratt-meanstats[selectt[ii],0])*np.ones((nn))/meanstats[selectt[ii],1]  ## FIXED REFERENCE NUMBER
                                    xx0[:,ii]=xtemp
                                    stringstemp=inputt_descrp[selectt[ii]]+'='+str(round(ref[selectt[ii]]*ratt,2))+';'
                                    strings0=strings0+stringstemp
                            if lnn==0:
                                xx=xx0
                                strings=strings0
                            else:
                                xx=np.vstack((xx,xx0))
                                strings=np.vstack((strings,strings0))
                                
                            if plotuncert==1:        
                                [pred,uncert]=predict_with_uncertainty(f, xx0, 1,1000)  ## dimension of output, number of iterations
                                pred=pred+minn
                                testpredl=pred-uncert
                                testpredu=pred+uncert
    #                            plt.errorbar(x3,pred[:,num]-correct, yerr=uncert, fmt='o')
                                ax.plot(x3,pred[:,num])
                                ax.fill_between(x3,testpredl[:,num],testpredu[:,num],alpha=0.35)    
                            else:
                                encoded3 = model.predict(xx)
                                
                                fig = plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
                                ax = fig.add_subplot(111)
                                plt.xlabel(inputt_descrp[sel], fontsize = 16)
                                plt.ylabel(inputt_descrp[selectrainout[num]], fontsize = 16)
                                plt.plot(x3,encoded3[:,num]+minn)
                        if importwave>1:
                            if selectt[npp]==6 or selectt[npp]==7:
                                ax.set_xscale('log')
                        plt.xlabel(inputt_descrp[sel], fontsize = 16)
                        plt.ylabel(inputt_descrp[selectrainout[num]], fontsize = 16)
                        plt.grid()
                        plt.xticks(fontsize=16)  
                        plt.yticks(fontsize=16)            
    #                    if full!=1:
                        plt.legend([istr[0] for istr in strings], fontsize = 16, loc = 'best')
                        if selectrainout[0]==0:
                            plt.ylim([0.3,.6])
                      
            """ BACKWARD ELIMINATION """
            if countitert==max(range(countiter)) and backward==1:
                strlabels=[i for i in range(len(selecttall)+1)]
                strlabels=np.array(strlabels) 
                strs = ["Input" for x in range(len(selecttall)+1)]
                strs[0]="Target"
                strs=np.array(strs)  
                for ii in range(len(mapeall)):
                    mapeall[ii]="{:.2f}".format(mapeall[ii])
                mapeallinput=np.hstack(("/",mapeall))
                allinput=np.vstack((input_descrp_totab.T,strs.T,mapeallinput))
                strcolumns=['Index','Variable','Role','MAPE \%']
                descrp=input_descrp_totab.T.tolist()
                descrp=descrp[0]
                my_dict={strcolumns[0]:strlabels.T.tolist(),strcolumns[1]:descrp,strcolumns[2]:strs.T.tolist(),strcolumns[3]:mapeallinput.tolist()}
                fig, ax = plt.subplots()
                ax.axis('off')
                ax.axis('tight')
                df = pd.DataFrame(my_dict, columns=strcolumns)    
                table=ax.table(cellText=df.values, colLabels=df.columns, loc='center')
                cells = table.properties()["celld"]
                for i in range(0, len(selectt)+3):
                    for j in range(0,len(strcolumns)):
                        cells[i, j]._loc = 'center'
                table.auto_set_font_size(False)      
                table.set_fontsize(30)
                table.scale(1.7,3)
                fig.tight_layout()
                plt.rc('font',family='Times New Roman')
                plt.show()    
                stringinp='string'+str(testnumall[0])
                plt.savefig(stringinp+'.png') 
            ## plot backward elimination
                mapeall=[6.85,12.74,14.13,10.75,6.7,6.7,7.0,6.9]
                mapeall=np.array(mapeall)
                fig = plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
                ax = fig.add_subplot(111)
                x=np.linspace(1,len(mapeall),len(mapeall),dtype=int)
                pk=np.argwhere(mapeall[:]==max(mapeall))
                
                wh=np.argwhere(mapeall[:]>1.20*mapeall[-1])      
                xwh=x[wh]
                mapeallwh=mapeall[wh]
                sm=np.argwhere(mapeall[:]<=mapeall[-1])      
                xsm=x[sm]
                mapeallsm=mapeall[sm]        
                plt.xticks(fontsize=16)  
                plt.yticks(fontsize=16)     
                ax.plot(x,mapeall,'o-',markersize=12)
                          
        #        ax.plot(x,mapeall,'o-',xwh,mapeallwh,'^',xsm,mapeallsm,'s',markersize=12)
                #    ax.fill_between(testout_data[:,num],testpredl[:,num],testpredu[:,num],color='black',alpha=0.6)      
                plt.xlabel('Index', fontsize = 20)
                plt.ylabel('MAPE \%', fontsize = 20)
                plt.grid()        
                xint = range(min(x), math.ceil(max(x))+1)
                plt.xticks(xint)  
        
                fig = plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
                ax = fig.add_subplot(111)
                x=np.linspace(1,len(mapeall),len(mapeall),dtype=int)
                pk=np.argwhere(mapeall[:]==max(mapeall))
                
                wh=np.argwhere(mapeall[:]>1.20*mapeall[-1])      
                xwh=x[wh]
                mapeallwh=mapeall[wh]
                sm=np.argwhere(mapeall[:]<=mapeall[-1])      
                xsm=x[sm]
                mapeallsm=mapeall[sm]        
                plt.xticks(fontsize=16)  
                plt.yticks(fontsize=16)     
        #        ax.plot(x,mapeall,'o-',markersize=12)
                          
                ax.plot(x,mapeall,'o-',x,mapetrainall,'s',markersize=12)
                plt.xlabel('Index', fontsize = 20)
                plt.ylabel('MAPE \%', fontsize = 20)
                plt.grid()        
                xint = range(min(x), math.ceil(max(x))+1)
                plt.xticks(xint)  
            
            
        #  ## ANALYZE DATA INPUT  
        #        ii=11                            
        #        fig = plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
        #        ax = fig.add_subplot(111)
        #        ax.plot(range(0,inputt.shape[0]),inputt[:,selecttall[ii]])
        #        #    ax.fill_between(testout_data[:,num],testpredl[:,num],testpredu[:,num],color='black',alpha=0.6)      
        #        plt.xlabel('Num', fontsize = 20)
        #        plt.ylabel('Param', fontsize = 20)
        #        plt.grid()        
        #        plt.legend([inputt_descrp[selecttall[ii]]], fontsize = 16)        
        
            
            #""" CONTOUR PLOTS"""
            #""" EFFECT OF cla,clv in cf and il dirs"""
            #
            if backward==0 and plotdetail==1 and len(selectt)>2 and len(selectt)<6:# and importwave<=1:      
        #selecttall0=[0,1,10,11,selectrainout[0],17,18,22]      
        #        num2plot=[1,3,0,2] # CF dirs """
        #        num2plot=[2,3,0,1] # CF dirs """
    #            num2plot=[0,1,2,3,4] # CF dirs """
                ref=[0.6,0.1,0.1,0.3,0.1,0.1,.1,.2,40000,6.6, 3.3, 90,90, 90, 0.6,0.8,3000,0,0.8,0.8,.5,1,0,1,0,0,0,1,0,0,0,1,0,1,0.0001,0.85,0.85,100,1300,0.01,0]  
                num2plot=list(range(0,len(selectt)))
                nn=100
                xxx=np.zeros((nn,nn))
                yyy=np.zeros((nn,nn))
                xxx_normin=np.zeros((nn,nn))
                yyy_normin=np.zeros((nn,nn))
                ooo=np.zeros((nn,nn))
                uuu=np.zeros((nn,nn))
                
                xx=np.zeros((nn,len(selectt)))
                
                ## xx axis
                ii=num2plot[0]
                sel=selectt[ii]-1
                for ii in range(0,len(xxx)):
                    xxx[:,ii]=np.linspace(min(inputt[:,sel+1])-0.00000001,max(inputt[:,sel+1])+0.00000000001,nn)
                    
                xxx_normin=(xxx-meanstats[sel+1,0])/meanstats[sel+1,1]  # ORIGINAL DATA
    
                ## yy axis
                ii=num2plot[1]    
                sel=selectt[ii]-1
                for ii in range(0,len(xxx)):
                    yyy[ii,:]=np.linspace(min(inputt[:,sel+1])-0.00000000001,max(inputt[:,sel+1])+0.000000000001,nn)
                    
                yyy_normin=(yyy-meanstats[sel+1,0])/meanstats[sel+1,1]  # ORIGINAL DATA
                
                
                wlegend=0
                
                if len(selectt)>3:
                    wlegend=1
                    for jj in range(2,len(num2plot)): ## THE REST VARIABLES ARE ZEROS
                        ii=num2plot[jj] 
                        xtemp=(ref[selectt[ii]]-meanstats[selectt[ii],0])*np.ones((nn))/meanstats[selectt[ii],1]
                        xx[:,ii]=xtemp
                        s3_meas1= np.argwhere(abs(inputt[:,selectt[ii]]-ref[selectt[ii]])<=0.2*ref[selectt[ii]]).tolist()
                        if jj==2:
                            s3_meas=s3_meas1
                        else:
                            s3_meas=intersection(s3_meas, s3_meas1)
        
                    
                if wlegend==1:
                    for ii in range(0,len(xxx)):
                        xx[:,num2plot[0]]=xxx_normin[:,ii]
                        xx[:,num2plot[1]]=yyy_normin[:,ii]
    #                    oout=model.predict(xx)
                        [pred,uncert]=predict_with_uncertainty(f, xx, 1,1000)  ## dimension of output, number of iterations
                        ooo[:,ii] = pred[:,num]
                        uuu[:,ii] = uncert[:,num]
                    strings2=''    
                    for jj in range(2,len(num2plot)):
                        strings2+=inputt_descrp[selectt[num2plot[jj]]]+'='+str(ref[selectt[num2plot[jj]]])+';'  
                        
                    fig = plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
                    ax = fig.add_subplot(111)                   
                    plt.xlabel(inputt_descrp[selectt[num2plot[0]]], fontsize = 16)
                    plt.ylabel(inputt_descrp[selectt[num2plot[1]]], fontsize = 16)
                    CS=plt.contourf(xxx,yyy,ooo+minn,label=[strings2])
                    plt.grid()
                    plt.xticks(fontsize=16)  
                    plt.yticks(fontsize=16)              
                    plt.gca().legend([strings2], fontsize = 16)
                    plt.colorbar(CS)
                    plt.plot(inputt[s3_meas,selectt[num2plot[0]]],inputt[s3_meas,selectt[num2plot[1]]],'ko')
                    if selectt[num2plot[0]]==6 or selectt[num2plot[0]]==7:
                        ax.set_xscale('log')
                    if selectt[num2plot[1]]==6 or selectt[num2plot[1]]==7:
                        ax.set_yscale('log')
                    fig = plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
                    ax = fig.add_subplot(111)                   
                    plt.xlabel(inputt_descrp[selectt[num2plot[0]]], fontsize = 16)
                    plt.ylabel(inputt_descrp[selectt[num2plot[1]]], fontsize = 16)
    #                CS=plt.contourf(xxx,yyy,uuu/(ooo+minn),label=[strings2])
                    CS=plt.contourf(xxx,yyy,uuu,label=[strings2])
                    plt.grid()
                    plt.xticks(fontsize=16)  
                    plt.yticks(fontsize=16)              
        #            view_colormap('jet')
                    plt.gca().legend([strings2], fontsize = 16)
                    plt.colorbar(CS)
                    if selectt[num2plot[0]]==6 or selectt[num2plot[0]]==7:
                        ax.set_xscale('log')
                    if selectt[num2plot[1]]==6 or selectt[num2plot[1]]==7:
                        ax.set_yscale('log')
                '''Not used afterwards'''
                if prunedata==1:
                    a=inputt_predictions
                    b=inputt[:,0]
    #                a=train_predictions
    #                b=trainout
    #                a=pred2
    #                b=inputttolabel[:,0]
                    ## uniform
                    errorto=np.zeros((len(a),1))
                    for i in range(0,a.shape[0]):
                        errorto[i]=abs(a[i]-b[i])
                    xerror=np.where(abs(errorto)>.10)
                    plt.plot(inputt[xerror[0],10],errorto[xerror[0]],'*')
                    plt.plot(inputt[xerror[0],18],errorto[xerror[0]],'*')
                    plt.plot(inputt[xerror[0],16],errorto[xerror[0]],'*')
    #                inputtpruned=inputt[]
                    inputtpruned=np.delete(inputt, xerror[0], axis = 0)
                    testnumpruned=308
                    sio.savemat('dataall'+str(testnumpruned)+'.mat', {'dataall':inputtpruned})
                    ## sheared flow
                    a=pred2
                    b=inputttolabel[:,0]
                    ## uniform
                    errorto=np.zeros((len(a),1))
                    for i in range(0,a.shape[0]):
                        errorto[i]=abs(a[i]-b[i])
                    xerror=np.where(abs(errorto)>.10)
                    plt.plot(inputttolabel[:,10],errorto,'o',inputttolabel[xerror[0],10],errorto[xerror[0]],'r*')
                    plt.plot(inputttolabel[:,18],errorto,'o',inputttolabel[xerror[0],18],errorto[xerror[0]],'*')
                    
                    fig = plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    #                ax = fig.add_subplot(111)   
                    ax = plt.axes(projection='3d')      
                    ax.view_init(90, 35)
    #                plt.xlabel(inputt_descrp[selectt[num2plot[10]]], fontsize = 16)
    #                plt.ylabel(inputt_descrp[selectt[num2plot[18]]], fontsize = 16)
                    ax.scatter3D(inputttolabel[:,10],inputttolabel[:,18],errorto,'*')
                    plt.grid()
                    plt.xticks(fontsize=16)  
                    plt.yticks(fontsize=16)              
        #            view_colormap('jet')
                    plt.gca().legend([strings2], fontsize = 16)
                    plt.colorbar(CS)
    #                 ax.scatter(inputttolabel[:,10], ys, zs)
    
    #        sio.savemat('dataall'+str(testnum)+'.mat', data)
    #        sio.savemat('traindata'+str(testnum)+'.mat', train)
    #        sio.savemat('testdata'+str(testnum)+'.mat', test)
    
            ## get the first layer weight
        #        if layernums[-1]==2:    
        #            count=1
        #            for layer in model.layers:
        #                weights = layer.get_weights() # list of numpy arrays
        #                encoder_weights=weights[0]
        #                encoder_bias=weights[1]                
        #                count=count+1
        #                auto_encoded=np.add(np.matmul(train_data_normin, encoder_weights), encoder_bias)
        #                if count<=len(layernums):
        #                    for i in range(auto_encoded.shape[0]):
        #                        for j in range(auto_encoded.shape[1]):
        #                            auto_encoded[i,j]=sigmoid(auto_encoded[i,j])
        #    
        #            x = auto_encoded[:, 0]
        #            labels=trainout_data
        #            c = labels
        #            s = 20  # size of points
        #            y = auto_encoded2[:, 1]
        #            for i in range(0,1,1):
        #        #    for i in range(0,train_data.shape[1],1):
        #                fig, ax = plt.subplots(num=None, figsize=(8, 6), dpi=100, facecolor='w', edgecolor='k')
        #                t = np.linspace(-1,1,50)
        #                im = plt.scatter(x, y, c=c, s=s, cmap=plt.cm.jet)
        #                plt.axis('equal')
        #                rangee=5
        #                plt.xlim([-rangee,rangee])
        #                plt.ylim([-rangee,rangee])
        #                plt.grid()
        #                plt.xticks(fontsize=16)  
        #                plt.yticks(fontsize=16)                          
        #                plt.xlabel('Latent feature 1', fontsize = 16)
        #                plt.ylabel('Latent feature 2', fontsize = 16)
        #                # Add a colorbar
        #                fig.colorbar(im, ax=ax)
postprocess=0
if pruning>0 and l1r>4:
    #l1rall=[0.0001,0.001,0.01,0.1,0.2,0.4,0.6,0.8]
    l1rall=[0.01,0.1,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5]
    l1rall=[0.0001,0.01,0.1,.11,.125,.15,.2,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5]
    full=1
    nnnum=2
    if postprocess==1:
        for itt in range(0,len(l1rall)):
            l1r=l1rall[itt]
            stringsavepenal='./currentanal'+str(nnnum)+str(testnumprint)+'full'+str(full)+'out'+str(selectrainout[0])+'l1r'+str(l1r)+'vscale'+str(vscalee)+'curlearn'+str(curlearn)
            
            data=pd.read_csv(stringsavepenal+'.csv')   
            data=data.values
            pair=np.array([data[0,0],data[0,1],l1r,data[0,3]])
            nweight=data[:,2].T
            if itt==0:
                pairs=pair
                nweights=nweight
            else:
                pairs=np.vstack((pairs,pair))
                nweights=np.vstack((nweights,nweight))
        nweights=nweights.T         
                
        fig = plt.figure(num=None, figsize=(8, 6), dpi=100, facecolor='w', edgecolor='k')
        ax = fig.add_subplot(111)
    #    for itt in range(0,len(l1rall)):
        for itt in range(0,8):
            neuronw=nweights[:,itt]
            plt.bar(range(1,1+len(neuronw)), neuronw/np.max(neuronw), align='center', alpha=0.5)
        ax.legend(['No'+str(l1rall[itt])], fontsize = 20)
        #plt.xticks(y_pos, objects)
        plt.ylabel('Weights', fontsize = 16)
        plt.xlabel('Variable index', fontsize = 16)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))   
        plt.xticks(fontsize=16)  
        plt.yticks(fontsize=16)
        plt.grid()
        plt.rc('font',family='Times New Roman')        
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)              
    #plt.plot(pairs[:,2],pairs[:,1],'o')
    #plt.plot(pairs[:,2],pairs[:,0],'o')        
    
    fig = plt.figure(num=None, figsize=(8, 6), dpi=100, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(111)
    plt.xlabel('Number of variables%', fontsize = 16)
    plt.ylabel('Prediction accuracy ', fontsize = 16)
    #ax.scatter(pairs[:,0],pairs[:,3],s=50)
    ax.scatter(pairs[:,0],pairs[:,1],s=50)
    
    plt.xticks(fontsize=16)  
    plt.yticks(fontsize=16)
    plt.grid()
    plt.rc('font',family='Times New Roman')        
    #        ax.legend(['Test data'], fontsize = 20)
    [len(np.argwhere((inputt[:,13]>3000) & (inputt[:,13]<3100))),len(np.argwhere((inputt[:,13]<3100))),len(np.argwhere((inputt[:,13]>4000) & (inputt[:,13]<4100))),len(np.argwhere((inputt[:,13]>4100) & (inputt[:,13]<5000))),len(np.argwhere((inputt[:,13]>5000) ))]
    
    plt.plot(traindata[:,6],trainlabel,'o')
    ######### CHECK FIGURE
    checkperturb=1
    css=[6,8]
    css=[6,51]
    css=[6,51]
    css=[6,8]
#    css=[51,51]
    if selectrainout[0]==5:
        
        inputt_descrp[selectrainout[0]]=r'$A*_{rms,cf3}/A*_{rms,cf}$'
    if checkperturb==1:
        if mapee>=6:
            
            tend=int(ttsize+(test_data.shape[0]-ttsize)/3*1.5)
            tend2=int(ttsize+(test_data.shape[0]-ttsize)/3*3)
        else:
            tend=int(ttsize+(test_data.shape[0]-ttsize)/3*1)
            tend2=int(ttsize+(test_data.shape[0]-ttsize)/3*2)
            tend3=int(ttsize+(test_data.shape[0]-ttsize)/3*3)
        fig = plt.figure(num=None, figsize=(8, 6), dpi=100, facecolor='w', edgecolor='k')
        ax = fig.add_subplot(111)
        plt.plot(testdata[0:ttsize,css[0]],testdata[0:ttsize,selectrainout[0]]+minn,'o')
        plt.plot(testdata[ttsize:tend,css[0]],testdata[ttsize:tend,selectrainout[0]]+minn,'+')
        plt.xlabel(inputt_descrp[css[0]], fontsize = 16)
        plt.ylabel(inputt_descrp[selectrainout[0]], fontsize = 16)
        plt.xticks(fontsize=16)  
        plt.yticks(fontsize=16)
        plt.grid()
        plt.rc('font',family='Times New Roman')
        if css[0]==6 or css[0]==7:
            ax.set_xscale('log')
        ax.legend(['Original data','Perturbed data'], fontsize = 20)
        plt.savefig(str(testnumprint)+str(selectrainout[0])+'perturbcstar.png', dpi=350) 
    #        ax.legend(['Test data'], fontsize = 20)
    
        fig = plt.figure(num=None, figsize=(8, 6), dpi=100, facecolor='w', edgecolor='k')
        ax = fig.add_subplot(111)
        plt.plot(testdata[0:ttsize,css[1]],testdata[0:ttsize,selectrainout[0]]+minn,'o')
        plt.plot(testdata[tend:tend2,css[1]],testdata[tend:tend2,selectrainout[0]]+minn,'+')
#        ax.set_xscale('log')
        plt.xlabel(inputt_descrp[css[1]], fontsize = 16)
        plt.ylabel(inputt_descrp[selectrainout[0]], fontsize = 16)
        plt.xticks(fontsize=16)  
        plt.yticks(fontsize=16)
        plt.grid()
        plt.rc('font',family='Times New Roman')     
        if css[1]==6 or css[1]==7:
            ax.set_xscale('log')
        ax.legend(['Original data','Perturbed data'], fontsize = 20)
        plt.savefig(str(testnumprint)+str(selectrainout[0])+'perturbre.png', dpi=350) 
#        fig = plt.figure(num=None, figsize=(8, 6), dpi=100, facecolor='w', edgecolor='k')
#        ax = fig.add_subplot(111)
#        plt.plot(testdata[0:ttsize,1],testdata[0:ttsize,0]+minn,'o')
#        plt.plot(testdata[tend2:-1,1],testdata[tend2:-1,0]+minn,'+')
#        plt.xlabel(inputt_descrp[1], fontsize = 16)
#        plt.ylabel(input_descrp_out, fontsize = 16)
#        plt.xticks(fontsize=16)  
#        plt.yticks(fontsize=16)
#        plt.grid()
#        plt.rc('font',family='Times New Roman')        
#        ax.legend(['Original data','Perturbed data'], fontsize = 20)
#        ax.set_xscale('log')
