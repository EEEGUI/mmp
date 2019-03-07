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
import sys
from utils.log import Logger
from datetime import datetime
sys.stdout = Logger("log.txt", sys.stdout)
os.urandom(2019)


def generate_feature(df, float_features):
    """
    生成新特征
    :return:
    """
    # Week
    first = datetime(2018, 1, 1)
    datedict2 = {}
    date_dict = np.load('data/raw/AvSigVersionTimestamps.npy')[()]
    for x in date_dict: datedict2[x] = (date_dict[x] - first).days // 7
    df['Week'] = df['AvSigVersion'].map(datedict2)
    float_features.append('Week')

    df['EngineVersion_2'] = df['EngineVersion'].apply(lambda x: x.split('.')[2]).astype(
        'category')
    df['EngineVersion_3'] = df['EngineVersion'].apply(lambda x: x.split('.')[3]).astype(
        'category')

    df['AppVersion_1'] = df['AppVersion'].apply(lambda x: x.split('.')[1]).astype('category')
    df['AppVersion_2'] = df['AppVersion'].apply(lambda x: x.split('.')[2]).astype('category')
    df['AppVersion_3'] = df['AppVersion'].apply(lambda x: x.split('.')[3]).astype('category')

    df['AvSigVersion_0'] = df['AvSigVersion'].apply(lambda x: x.split('.')[0]).astype('category')
    df['AvSigVersion_1'] = df['AvSigVersion'].apply(lambda x: x.split('.')[1]).astype('category')
    df['AvSigVersion_2'] = df['AvSigVersion'].apply(lambda x: x.split('.')[2]).astype('category')
    # self.df_all['OsBuildLab_0'] = self.df_all['OsBuildLab'].astype('str').apply(lambda x: x.split('.')[0]).astype('category')
    # self.df_all['OsBuildLab_1'] = self.df_all['OsBuildLab'].astype('str').apply(lambda x: x.split('.')[1]).astype('category')
    # self.df_all['OsBuildLab_2'] = self.df_all['OsBuildLab'].astype('str').apply(lambda x: x.split('.')[2]).astype('category')
    # self.df_all['OsBuildLab_3'] = self.df_all['OsBuildLab'].astype('str').apply(lambda x: x.split('.')[3]).astype('category')
    # self.df_all['OsBuildLab_40'] = self.df_all['OsBuildLab'].apply(lambda x: x.split('.')[4].split('-')[0]).astype('category')
    # self.df_all['OsBuildLab_41'] = self.df_all['OsBuildLab'].apply(lambda x: x.split('.')[4].split('-')[1]).astype('category')

    df['Census_OSVersion_0'] = df['Census_OSVersion'].apply(lambda x: x.split('.')[0]).astype(
        'category')
    df['Census_OSVersion_1'] = df['Census_OSVersion'].apply(lambda x: x.split('.')[1]).astype(
        'category')
    df['Census_OSVersion_2'] = df['Census_OSVersion'].apply(lambda x: x.split('.')[2]).astype(
        'category')
    df['Census_OSVersion_3'] = df['Census_OSVersion'].apply(lambda x: x.split('.')[3]).astype(
        'category')

    # https://www.kaggle.com/adityaecdrid/simple-feature-engineering-xd
    df['primary_drive_c_ratio'] = df['Census_SystemVolumeTotalCapacity'] / df[
        'Census_PrimaryDiskTotalCapacity']
    float_features += ['primary_drive_c_ratio']

    df['non_primary_drive_MB'] = df['Census_PrimaryDiskTotalCapacity'] - df[
        'Census_SystemVolumeTotalCapacity']
    float_features += ['non_primary_drive_MB']

    df['aspect_ratio'] = df['Census_InternalPrimaryDisplayResolutionHorizontal'] / df[
        'Census_InternalPrimaryDisplayResolutionVertical']
    float_features += ['aspect_ratio']

    df['monitor_dims'] = df['Census_InternalPrimaryDisplayResolutionHorizontal'].astype(
        str) + '*' + df['Census_InternalPrimaryDisplayResolutionVertical'].astype('str')

    df['monitor_dims'] = df['monitor_dims'].astype('category')

    df['dpi'] = ((df['Census_InternalPrimaryDisplayResolutionHorizontal'] ** 2 + df[
        'Census_InternalPrimaryDisplayResolutionVertical'] ** 2) ** .5) / (
                             df['Census_InternalPrimaryDiagonalDisplaySizeInInches'])
    float_features += ['dpi']

    df['dpi_square'] = df['dpi'] ** 2
    float_features += ['dpi_square']

    df['MegaPixels'] = (df['Census_InternalPrimaryDisplayResolutionHorizontal'] * df[
        'Census_InternalPrimaryDisplayResolutionVertical']) / 1e6
    float_features += ['MegaPixels']

    df['Screen_Area'] = (df['aspect_ratio'] * (
            df['Census_InternalPrimaryDiagonalDisplaySizeInInches'] ** 2)) / (
                                       df['aspect_ratio'] ** 2 + 1)
    float_features += ['Screen_Area']

    df['ram_per_processor'] = df['Census_TotalPhysicalRAM'] / df[
        'Census_ProcessorCoreCount']
    float_features += ['ram_per_processor']

    df['new_num_0'] = df['Census_InternalPrimaryDiagonalDisplaySizeInInches'] / df[
        'Census_ProcessorCoreCount']
    float_features += ['new_num_0']

    df['new_num_1'] = df['Census_ProcessorCoreCount'] * df[
        'Census_InternalPrimaryDiagonalDisplaySizeInInches']
    float_features += ['new_num_1']

    df['Census_IsFlightingInternal'] = df['Census_IsFlightingInternal'].fillna(1)
    df['Census_ThresholdOptIn'] = df['Census_ThresholdOptIn'].fillna(1)
    df['Census_IsWIMBootEnabled'] = df['Census_IsWIMBootEnabled'].fillna(1)
    df['Wdft_IsGamer'] = df['Wdft_IsGamer'].fillna(0)

    return df, float_features


print('Loading Train and Test Data.\n')
mmp_config = config.Config()
print('Reading train.h5...')
train = pd.read_hdf(mmp_config.TRAIN_H5_PATH, key='data')
print('Reading test.h5...')
test = pd.read_hdf(mmp_config.TEST_H5_PATH, key='data')

train['MachineIdentifier'] = train.index.astype('uint32')
test['MachineIdentifier'] = test.index.astype('uint32')
test['HasDetections'] = [0]*len(test)

float_features=['Census_SystemVolumeTotalCapacity','Census_PrimaryDiskTotalCapacity']

train, _ = generate_feature(train, float_features)
test, float_features = generate_feature(test, float_features)



# In[4]:


def make_bucket(data,num=10):
    data.sort()
    bins=[]
    for i in range(num):
        bins.append(data[int(len(data)*(i+1)//num)-1])
    return bins


for f in float_features:
    train[f] = train[f].fillna(1e10)
    test[f]=test[f].fillna(1e10)
    data=list(train[f])+list(test[f])
    bins=make_bucket(data,num=50)
    train[f]=np.digitize(train[f],bins=bins)
    test[f]=np.digitize(test[f],bins=bins)
    
train, dev, _, _ = train_test_split(train,train['HasDetections'],test_size=0.02, random_state=2019)
features = list(set(train.columns) - set(['HasDetections', 'MachineIdentifier']))


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
            batch_size=512, # 1024
            optimizer="adam",
            learning_rate=0.001,
            num_display_steps=250,
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
        random.seed(2019)
        tmp=random.sample(index,int(1.0/hparam.kfold*train.shape[0]))
    index=index-set(tmp)
    print("Number:",len(tmp))
    K_fold.append(tmp)
    

for i in range(hparam.kfold):
    print("Fold",i)
    dev_index=K_fold[i]
    random.seed(2019)
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
submission.to_csv('xdeepfm_submission(new_features).csv', index=False)

