#!/usr/bin/env python
# coding: utf-8

# # Download repo from https://github.com/guoday/ctrNet-tool

# In[1]:

import ctrNet
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from src import misc_utils as utils
import os
import gc
import random
from utils import config

print('Loading Train and Test Data.\n')
mmp_config = config.Config()
print('Reading train.h5...')
train = pd.read_hdf(mmp_config.TRAIN_H5_PATH, key='data')
print('Reading test.h5...')
test = pd.read_hdf(mmp_config.TEST_H5_PATH, key='data')

train['MachineIdentifier'] = train.index.astype('uint32')
test['MachineIdentifier']  = test.index.astype('uint32')
test['HasDetections']=[0]*len(test)


# In[4]:


def make_bucket(data,num=10):
    data.sort()
    bins=[]
    for i in range(num):
        bins.append(data[int(len(data)*(i+1)//num)-1])
    return bins
float_features=['Census_SystemVolumeTotalCapacity','Census_PrimaryDiskTotalCapacity']

for f in float_features:
    train[f]=train[f].fillna(1e10)
    test[f]=test[f].fillna(1e10)
    data=list(train[f])+list(test[f])
    bins=make_bucket(data,num=50)
    train[f]=np.digitize(train[f],bins=bins)
    test[f]=np.digitize(test[f],bins=bins)
    
train, dev,_,_ = train_test_split(train,train['HasDetections'],test_size=0.02, random_state=2019)
features=train.columns.tolist()[1:-1]


# # Creating hparams

# In[5]:


hparam=tf.contrib.training.HParams(
            model='xdeepfm',
            norm=True,
            batch_norm_decay=0.9,
            hidden_size=[128,128],
            cross_layer_sizes=[128,128,128],
            k=8,
            hash_ids=int(2e5),
            batch_size=1024,
            optimizer="adam",
            learning_rate=0.001,
            num_display_steps=1000,
            num_eval_steps=1000,
            epoch=1,
            metric='auc',
            activation=['relu','relu','relu'],
            cross_activation='identity',
            init_method='uniform',
            init_value=0.1,
            feature_nums=len(features),
            kfold=5)
utils.print_hparams(hparam)


# # Training model

# In[6]:


index=set(range(train.shape[0]))
K_fold=[]
for i in range(hparam.kfold):
    if i == hparam.kfold-1:
        tmp=index
    else:
        tmp=random.sample(index,int(1.0/hparam.kfold*train.shape[0]))
    index=index-set(tmp)
    print("Number:",len(tmp))
    K_fold.append(tmp)
    

for i in range(hparam.kfold):
    print("Fold",i)
    dev_index=K_fold[i]
    dev_index=random.sample(dev_index,int(0.1*len(dev_index)))
    train_index=[]
    for j in range(hparam.kfold):
        if j!=i:
            train_index+=K_fold[j]
    model=ctrNet.build_model(hparam)
    model.train(train_data=(train.iloc[train_index][features],train.iloc[train_index]['HasDetections']),
                dev_data=(train.iloc[dev_index][features],train.iloc[dev_index]['HasDetections']))
    print("Training Done! Inference...")
    if i==0:
        preds=model.infer(dev_data=(test[features],test['HasDetections']))/hparam.kfold
    else:
        preds+=model.infer(dev_data=(test[features],test['HasDetections']))/hparam.kfold


# # Inference

# In[7]:


submission = pd.read_csv('./data/raw/sample_submission.csv', nrows=len(preds))
submission['HasDetections'] = preds
print(submission['HasDetections'].head())
submission.to_csv('xdeepfm_submission.csv', index=False)

