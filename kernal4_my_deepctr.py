# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 14:13:19 2019

@author: koukoumumu
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score
from deepctr.models import DeepFM,xDeepFM
from deepctr import SingleFeat
from utils import config


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
print('Loading Train and Test Data.\n')
mmp_config = config.Config()
print('Reading train.h5...')
train = pd.read_hdf(mmp_config.TRAIN_H5_PATH, key='data')

print('Reading test.h5...')
test = pd.read_hdf(mmp_config.TEST_H5_PATH, key='data')

# train = pd.read_csv('data/raw/train.csv', dtype=dtypes, low_memory=True)
train['MachineIdentifier'] = train.index.astype('uint32')
train_size = train.shape[0]
# test  = pd.read_csv('data/raw/test.csv', dtype=dtypes, low_memory=True)
test['MachineIdentifier']  = test.index.astype('uint32')
test['HasDetections']=[0]*len(test)
data = pd.concat([train,test])

dense_features = ['Census_SystemVolumeTotalCapacity','Census_PrimaryDiskTotalCapacity']
sparse_features = [col for col in train.columns.tolist()[1:-1] if col not in dense_features]

for col in sparse_features:
    if str(data[col].dtype)=='category': # category类别的特征在填充空缺值时需要注意
        data[col] = data[col].cat.add_categories(['-1'])
        data[col] = data[col].fillna('-1')
    else:
        data[col] = data[col].fillna('-1')
data[dense_features] = data[dense_features].fillna(data[dense_features].mean(),)
target = ['HasDetections']

# 1.Label Encoding for sparse features,and do simple Transformation for dense features
for feat in sparse_features:
    lbe = LabelEncoder()
    data[feat] = data[feat].astype('str')
    data[feat] = lbe.fit_transform(data[feat])

#切除数值型特征的异常值,5%~95%
clip_05 = data[dense_features].quantile(.05)
clip_95 = data[dense_features].quantile(.95)
for col in dense_features:
    data[col] = data[col].clip(clip_05[col], clip_95[col])
    
mms = MinMaxScaler(feature_range=(0, 1))
data[dense_features] = mms.fit_transform(data[dense_features])

# 2.count #unique features for each sparse field,and record dense feature field name
sparse_feature_list = [SingleFeat(feat, data[feat].nunique()) for feat in sparse_features]
dense_feature_list = [SingleFeat(feat, 0) for feat in dense_features]

# 3.generate input data for model

train = data.iloc[:train_size]
test = data.iloc[train_size:]

train_model_input = [train[feat.name].values for feat in sparse_feature_list] + \
    [train[feat.name].values for feat in dense_feature_list]
test_model_input = [test[feat.name].values for feat in sparse_feature_list] + \
    [test[feat.name].values for feat in dense_feature_list]

# 4.Define Model,train,predict and evaluate
model = xDeepFM({"sparse": sparse_feature_list,
                "dense": dense_feature_list}, final_activation='sigmoid')
model.compile("adam", "binary_crossentropy",
              metrics=['binary_crossentropy'], )
# 4096
# 2**19
history = model.fit(train_model_input, train[target].values,
                    batch_size=1024, epochs=5, verbose=2, validation_split=0.2, )
pred_ans = model.predict(test_model_input, batch_size=2**10)
#print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
#print("test AUC", round(roc_auc_score(train[target].values, pred_ans), 4))

submission = pd.read_csv('data/raw/sample_submission.csv')
submission['HasDetections'] = pred_ans
#print(submission['HasDetections'].head())
submission.to_csv('nffm_submission.csv', index=False)