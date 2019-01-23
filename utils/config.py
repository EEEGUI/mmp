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

    ##  DataSet ##
    LABEL_COL_NAME = 'HasDetections'
    NROWS = None
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
    PARAM = {'num_leaves': 60,
             'min_data_in_leaf': 60,
             'objective': 'binary',
             'max_depth': -1,
             'learning_rate': 0.1,
             "boosting": "gbdt",
             "feature_fraction": 0.8,
             "bagging_freq": 1,
             "bagging_fraction": 0.8,
             "bagging_seed": 11,
             "metric": 'auc',
             "lambda_l1": 0.1,
             "random_state": 133,
             "verbosity": -1}
    N_FOLDS = 5
    NUM_BOOST_ROUND = 10000
    EARLY_STOP_ROUND = 100
    MODEL_SAVING_PATH = 'assets/model.pkl'




