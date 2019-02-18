import numpy as np


class Config(object):

    ##  File Path ##

    TRAIN_PATH = 'data/raw/train.csv'
    TEST_PATH = 'data/raw/test.csv'

    TRAIN_H5_PATH = 'data/raw/train.h5'
    TEST_H5_PATH = 'data/raw/test.h5'

    TRAIN_FEATURE_PATH = 'data/input/train_feature.h5'
    TEST_FEATURE_PATH = 'data/input/test_feature.h5'
    LABEL_PATH = 'data/input/label.h5'
    OUTPUT = 'data/output'

    TRAIN_REPORT_PATH = 'assets/train_report.html'
    TEST_REPORT_PATH = 'assets/test_report.html'

    FEATURE_IMPORTANCE_FIG = 'assets/feature_importance.png'

    FEATURE_TO_DROP_JSON = 'assets/features_to_drop.json'

    LIGHTGBM_BEST_PARAM = 'assets/lightgbm_param.json'

    VERSION_TIME_DICT_PATH = 'data/raw/AvSigVersionTimestamps.npy'

    ##  DataSet ##
    LABEL_COL_NAME = 'HasDetections'
    NROWS = 10000
    RANDOM_SAMPLE_PERCENTAGE = 1  # 训练集使用比例
    KEY = 'MachineIdentifier'
    DTYPES = {
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

    NUMBER_TYPE = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']


    ## Feature Engineer ##

    TRUE_NUMERICAL_COLUMNS = [
        'Census_ProcessorCoreCount',
        'Census_PrimaryDiskTotalCapacity',
        'Census_SystemVolumeTotalCapacity',
        'Census_TotalPhysicalRAM',
        'Census_InternalPrimaryDiagonalDisplaySizeInInches',
        'Census_InternalPrimaryDisplayResolutionHorizontal',
        'Census_InternalPrimaryDisplayResolutionVertical',
        'Census_InternalBatteryNumberOfCharges'
    ]

    FREQUENT_ENCODED_COLUMNS = ['AppVersion',
                                'AVProductStatesIdentifier',
                                'AvSigVersion',
                                'Census_ChassisTypeName',
                                'Census_FirmwareManufacturerIdentifier',
                                'Census_FirmwareVersionIdentifier',
                                'Census_InternalBatteryType',
                                'Census_OEMModelIdentifier',
                                'Census_OEMNameIdentifier',
                                'Census_OSBuildRevision',
                                'Census_OSVersion',
                                'Census_ProcessorModelIdentifier',
                                'CityIdentifier',
                                'CountryIdentifier',
                                'DefaultBrowsersIdentifier',
                                'EngineVersion',
                                'GeoNameIdentifier',
                                'IeVerIdentifier',
                                'LocaleEnglishNameIdentifier',
                                'OsBuildLab',
                                'OsBuild',
                                'OsVer']

    COLUMNS_TO_DROP = ['AutoSampleOptIn', 'Census_IsFlightingInternal', 'Census_ProcessorClass']

    COLUMNS_TO_SPLIT = ['AvSigVersion', 'AppVersion', 'Census_OSVersion', 'EngineVersion', 'OsVer']
    ## Model ##

    # PARAM = {'num_leaves': 60,
    #          'min_data_in_leaf': 60,
    #          'objective': 'binary',
    #          'max_depth': -1,
    #          'learning_rate': 0.1,
    #          "boosting": "gbdt",
    #          "feature_fraction": 0.8,
    #          "bagging_freq": 1,
    #          "bagging_fraction": 0.8,
    #          "bagging_seed": 11,
    #          "metric": 'auc',
    #          "lambda_l1": 0.1,
    #          "random_state": 133,
    #          "verbosity": -1}

    # 0.689
    PARAM1 = {
                'boosting_type': 'gbdt',
                'objective': 'binary',
                'metric': 'auc',
                'learning_rate': 0.05,
                'max_depth': 5,
                'num_leaves': 20,
                'sub_feature': 0.9,
                'sub_row': 0.9,
                'bagging_freq': 1,
                'lambda_l1': 0.1,
                'lambda_l2': 0.1,
                'random_state': 133,
                'verbosity': -1,
                'score': 0.5,
                'num_boost_round': 10000,
                'early_stopping_rounds': 200
             }

    #
    PARAM = {
                'boosting_type': 'gbdt',
                'objective': 'binary',
                'metric': 'auc',
                'learning_rate': 0.05,
                'max_depth': -1,
                'num_leaves': 2**12-1,
                'sub_feature': 0.28,
                'sub_row': 0.8,
                'bagging_freq': 1,
                'lambda_l1': 0.2,
                'lambda_l2': 0.2,
                'random_state': 133,
                'verbosity': -1,
                'score': 0.5,
                'num_boost_round': 30000,
                'early_stopping_rounds': 100
             }

    # 0.693
    PARAM2 = {
                            'max_depth': -1,
                            'metric': 'auc',
                            'n_estimators': 30000,
                            'learning_rate': 0.05,
                            'num_leaves': 2**12-1,
                            'colsample_bytree': 0.28,
                            'objective': 'binary',
                            'n_jobs': -1,
                            'early_stopping_rounds': 100,
                            'verbosity': -1
                            }

    ## Param Search ##

    PARAM_GRID = {
        'boosting_type': ['gbdt', 'goss', 'dart'],
        'objective': ['binary'],
        'metric': ['auc'],
        'num_leaves': list(range(30, 150)),
        'learning_rate': list(np.logspace(np.log10(0.005), np.log10(0.5), base=10, num=1000)),
        'min_child_samples': list(range(10, 100)), # default 20
        'reg_alpha': list(np.linspace(0, 1)), # default 0 正则L1
        'reg_lambda': list(np.linspace(0, 1)), # default 0 正则L2
        'feature_fraction': list(np.linspace(0.6, 1, 10)), # 特征抽取 default 1.0
        'bagging_fraction': list(np.linspace(0.5, 1, 100)), # 数据抽取 default 1.0
        'num_boost_round': [10000],
        'early_stopping_rounds': [200]
    }

    SEARCH_TIME = 5
    RANDOM_STATE = 712
    N_FOLDS = 5
    MODEL_SAVING_PATH = 'assets/model.pkl'




