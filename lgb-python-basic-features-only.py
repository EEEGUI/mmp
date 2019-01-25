#!/usr/bin/env python
# coding: utf-8

# In[ ]:
from utils.config import Config
mmp_config = Config()
import numpy as np 
from numba import jit
import pandas as pd 
from datetime import datetime as dt
import os
import seaborn as sns 
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import lightgbm as lgb
import time
import datetime
from tqdm import tqdm_notebook as tqdm
from sklearn.model_selection import StratifiedKFold, KFold, TimeSeriesSplit
from sklearn import metrics
import gc

import warnings
warnings.filterwarnings("ignore")

import logging

logging.basicConfig(filename='log.txt',level=logging.DEBUG, format='%(asctime)s %(message)s')

pd.set_option('max_colwidth', 500)
pd.set_option('max_columns', 500)
pd.set_option('max_rows', 100)


# In[ ]:


@jit
def fast_auc(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    nfalse = 0
    auc = 0
    n = len(y_true)
    for i in range(n):
        y_i = y_true[i]
        nfalse += (1 - y_i)
        auc += y_i * nfalse
    auc /= (nfalse * (n - nfalse))
    return auc


def eval_auc(preds, dtrain):
    labels = dtrain.get_label()
    return 'auc', fast_auc(labels, preds), True


# In[ ]:


#https://www.kaggle.com/theoviel/load-the-totality-of-the-data
dtypes = {
        'MachineIdentifier':                                    'category',
        'ProductName':                                          'category',
        'EngineVersion':                                        'category',
        'AppVersion':                                           'category',
        'AvSigVersion':                                         'category',
        'IsBeta':                                               'int8',
        'RtpStateBitfield':                                     'float16',
        'IsSxsPassiveMode':                                     'int8',
        'DefaultBrowsersIdentifier':                            'float16',
        'AVProductStatesIdentifier':                            'float32',
        'AVProductsInstalled':                                  'float16',
        'AVProductsEnabled':                                    'float16',
        'HasTpm':                                               'int8',
        'CountryIdentifier':                                    'int16',
        'CityIdentifier':                                       'float32',
        'OrganizationIdentifier':                               'float16',
        'GeoNameIdentifier':                                    'float16',
        'LocaleEnglishNameIdentifier':                          'int8',
        'Platform':                                             'category',
        'Processor':                                            'category',
        'OsVer':                                                'category',
        'OsBuild':                                              'int16',
        'OsSuite':                                              'int16',
        'OsPlatformSubRelease':                                 'category',
        'OsBuildLab':                                           'category',
        'SkuEdition':                                           'category',
        'IsProtected':                                          'float16',
        'AutoSampleOptIn':                                      'int8',
        'PuaMode':                                              'category',
        'SMode':                                                'float16',
        'IeVerIdentifier':                                      'float16',
        'SmartScreen':                                          'category',
        'Firewall':                                             'float16',
        'UacLuaenable':                                         'float32',
        'Census_MDC2FormFactor':                                'category',
        'Census_DeviceFamily':                                  'category',
        'Census_OEMNameIdentifier':                             'float16',
        'Census_OEMModelIdentifier':                            'float32',
        'Census_ProcessorCoreCount':                            'float16',
        'Census_ProcessorManufacturerIdentifier':               'float16',
        'Census_ProcessorModelIdentifier':                      'float16',
        'Census_ProcessorClass':                                'category',
        'Census_PrimaryDiskTotalCapacity':                      'float32',
        'Census_PrimaryDiskTypeName':                           'category',
        'Census_SystemVolumeTotalCapacity':                     'float32',
        'Census_HasOpticalDiskDrive':                           'int8',
        'Census_TotalPhysicalRAM':                              'float32',
        'Census_ChassisTypeName':                               'category',
        'Census_InternalPrimaryDiagonalDisplaySizeInInches':    'float16',
        'Census_InternalPrimaryDisplayResolutionHorizontal':    'float16',
        'Census_InternalPrimaryDisplayResolutionVertical':      'float16',
        'Census_PowerPlatformRoleName':                         'category',
        'Census_InternalBatteryType':                           'category',
        'Census_InternalBatteryNumberOfCharges':                'float32',
        'Census_OSVersion':                                     'category',
        'Census_OSArchitecture':                                'category',
        'Census_OSBranch':                                      'category',
        'Census_OSBuildNumber':                                 'int16',
        'Census_OSBuildRevision':                               'int32',
        'Census_OSEdition':                                     'category',
        'Census_OSSkuName':                                     'category',
        'Census_OSInstallTypeName':                             'category',
        'Census_OSInstallLanguageIdentifier':                   'float16',
        'Census_OSUILocaleIdentifier':                          'int16',
        'Census_OSWUAutoUpdateOptionsName':                     'category',
        'Census_IsPortableOperatingSystem':                     'int8',
        'Census_GenuineStateName':                              'category',
        'Census_ActivationChannel':                             'category',
        'Census_IsFlightingInternal':                           'float16',
        'Census_IsFlightsDisabled':                             'float16',
        'Census_FlightRing':                                    'category',
        'Census_ThresholdOptIn':                                'float16',
        'Census_FirmwareManufacturerIdentifier':                'float16',
        'Census_FirmwareVersionIdentifier':                     'float32',
        'Census_IsSecureBootEnabled':                           'int8',
        'Census_IsWIMBootEnabled':                              'float16',
        'Census_IsVirtualDevice':                               'float16',
        'Census_IsTouchEnabled':                                'int8',
        'Census_IsPenCapable':                                  'int8',
        'Census_IsAlwaysOnAlwaysConnectedCapable':              'float16',
        'Wdft_IsGamer':                                         'float16',
        'Wdft_RegionIdentifier':                                'float16',
        'HasDetections':                                        'int8'
        }


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage(deep=True).sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


# In[ ]:


numerics = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerical_columns = [c for c,v in dtypes.items() if v in numerics]
categorical_columns = [c for c,v in dtypes.items() if v not in numerics]


# In[ ]:


train = pd.read_csv(mmp_config.TRAIN_PATH, dtype=dtypes, nrows=mmp_config.NROWS)


# In[ ]:


random_sample_percent = 1
random_state = 15
number_of_folds = 5
stop_after_fold_number = 1
shuffle = True


# In[ ]:


# To match the R kernel - https://www.kaggle.com/hung96ad/lightgbm
if random_sample_percent is not None:
    train = train.sample(frac=random_sample_percent, random_state=random_state)


# In[ ]:


train_y = train['HasDetections']


# In[ ]:


train = reduce_mem_usage(train)


# In[ ]:


test = pd.read_csv(mmp_config.TEST_PATH, dtype=dtypes, nrows=mmp_config.NROWS)
# test.loc[6529507, 'OsBuildLab'] = '17134.1.amd64fre.rs4_release.180410-1804'
test = reduce_mem_usage(test)


# In[ ]:


train.head()


# In[ ]:


def encode_categorical_columns(x_train, x_test, columns, sort=True):
    train_length = x_train.shape[0]
    for col in tqdm(columns):
        if col == 'MachineIdentifier' or col == 'HasDetections':
            continue
            
        combined_data = pd.concat([x_train[col], x_test[col]])
        combined_data, _ = pd.factorize(combined_data, sort=sort)
        combined_data = pd.Series(combined_data).astype('int32')
        x_train[col] = combined_data.iloc[:train_length].values
        x_test[col] = combined_data.iloc[train_length:].values
        x_train[col] = x_train[col].fillna(0)
        x_test[col] = x_test[col].fillna(0)
        del combined_data
        
    return x_train, x_test


# In[ ]:


train, test = encode_categorical_columns(train, test, categorical_columns)
train = reduce_mem_usage(train)
test = reduce_mem_usage(test)

# In[ ]:


def train_model(x, y, lgb_params, 
                number_of_folds=5, 
                evaluation_metric='auc', 
                save_feature_importances=False, 
                early_stopping_rounds=50, 
                num_round = 50,
                identifier_columns=['MachineIdentifier'],
                stop_after_fold_number=None):
    cross_validator = StratifiedKFold(n_splits=number_of_folds,
                                  random_state=random_state,
                                  shuffle=shuffle)
    
    validation_scores = []
    classifier_models = []
    feature_importance_df = pd.DataFrame()
    for fold_index, (train_index, validation_index) in enumerate(cross_validator.split(x, y)):
        x_train, x_validation = x.iloc[train_index], x.iloc[validation_index]
        y_train, y_validation = y.iloc[train_index], y.iloc[validation_index]
    
        x_train.drop(identifier_columns, axis=1, inplace=True)
        validation_identifier_data = x_validation[identifier_columns]
        x_validation.drop(identifier_columns, axis=1, inplace=True)
        x_train_columns = x_train.columns
        trn_data = lgb.Dataset(x_train,
                       label=y_train,
                       # categorical_feature=categorical_columns
                       )
        del x_train
        del y_train
        val_data = lgb.Dataset(x_validation,
                               label=y_validation,
                               # categorical_feature=categorical_columns
                               )
        classifier_model = lgb.train(lgb_params,
                                     trn_data,
                                      num_round,
                                     valid_sets=[trn_data, val_data],
                                     verbose_eval=100,
                                     early_stopping_rounds=early_stopping_rounds,
                                     feval=eval_auc
                                     )

        classifier_models.append(classifier_model)
        
        predictions = classifier_model.predict(x_validation, num_iteration=classifier_model.best_iteration)
        false_positive_rate, recall, thresholds = metrics.roc_curve(y_validation, predictions)
        score = metrics.auc(false_positive_rate, recall)
        validation_scores.append(score)
        
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = x_train_columns
        fold_importance_df["importance"] = classifier_model.feature_importance(importance_type='gain')
        fold_importance_df["fold"] = fold_index + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        if stop_after_fold_number == fold_index + 1:
            break
    if save_feature_importances:
        cols = (feature_importance_df[["feature", "importance"]]
                .groupby("feature")
                .mean()
                .sort_values(by="importance", ascending=False)[:1000].index)

        best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]

        plt.figure(figsize=(14, 25))
        sns.barplot(x="importance",
                    y="feature",
                    data=best_features.sort_values(by="importance",
                                                   ascending=False))
        plt.title('LightGBM Features (avg over folds)')
        plt.tight_layout()
        plt.savefig('lgbm_importances.png')

        # mean_gain = feature_importances[['gain', 'feature']].groupby('feature').mean()
        # feature_importances['mean_gain'] = feature_importances['feature'].map(mean_gain['gain'])
        #
        # temp = feature_importances.sort_values('mean_gain', ascending=False)
        best_features.sort_values(by="importance", ascending=False)\
                     .groupby("feature")\
                     .mean()\
                     .sort_values(by="importance", ascending=False)\
                     .to_csv('feature_importances_new.csv', index=True)

    score = sum(validation_scores) / len(validation_scores)
    return classifier_models, score


# In[ ]:


# # params from https://www.kaggle.com/fabiendaniel/detecting-malwares-with-lgbm
# params = {'num_leaves': 128,
#          'min_data_in_leaf': 42,
#          'objective': 'binary',
#          'max_depth': -1,
#          'learning_rate': 0.05,
#          "boosting": "gbdt",
#          "feature_fraction": 0.8,
#          "bagging_freq": 5,
#          "bagging_fraction": 0.8,
#          "bagging_seed": 11,
#          "lambda_l1": 0.15,
#          "lambda_l2": 0.15,
#          "random_state": 42,          
#          "verbosity": -1}

base_params = {   
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'nthread': 4,
        'learning_rate': 0.05,
        'max_depth': 5,
        'num_leaves': 40,
        'sub_feature': 0.9,
        'sub_row': 0.9,
        'bagging_freq': 1,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'random_state': random_state,
        'verbosity': -1,
        }
# base_params = {'objective':'binary', 
#                "boosting": "gbdt", 
#                'learning_rate': 0.02, 
#                'max_depth': -1,
#                "feature_fraction": 0.8, 
#                "bagging_freq": 1, 
#                "bagging_fraction": 0.8 , 
#                "bagging_seed": 11,
#                "metric": 'auc', 
#                "lambda_l1": 0.1, 
#                'num_leaves': 60, 
#                'min_data_in_leaf': 60, 
#                "verbosity": -1, 
#                "random_state": random_state
#               }


# In[ ]:

print('Shape of train is ', train.shape)

models, validation_score = train_model(train.drop('HasDetections', axis=1),
                                      train_y, base_params,
                                      num_round=5120,
                                       early_stopping_rounds=200,
                                      stop_after_fold_number=stop_after_fold_number,
                                      save_feature_importances=True)
del train


# In[ ]:


submission_data = test[['MachineIdentifier']]
predictions = np.zeros(len(test))
test = test.drop('MachineIdentifier', axis=1)
chunk_size = 1000000
for classifier_model in tqdm(models):
    current_pred = np.zeros(len(test))
    initial_idx = 0
    while initial_idx < test.shape[0]:
        final_idx = min(initial_idx + chunk_size, test.shape[0])
        idx = range(initial_idx, final_idx)
        current_pred[idx] = classifier_model.predict(test.iloc[idx],
                                                     num_iteration=classifier_model.best_iteration)
        initial_idx = final_idx

    predictions += current_pred / len(models)
del test


# In[ ]:


submission_data['HasDetections'] = predictions
filename = 'submission_{:.6f}_{}_folds_{}_data.csv'.format(validation_score,
                                                              dt.now().strftime('%Y-%m-%d-%H-%M'),
                                                              len(models))
submission_data.to_csv('single_{}'.format(filename), index=False)

