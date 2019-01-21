import pandas as pd
from utils import config
from utils import dataset
from utils.utils import timer
import numpy as np
from tqdm import tqdm


class MMPDataSet(dataset.DataSet):
    def __init__(self, df_train, df_test, config):
        super(MMPDataSet, self).__init__(df_train, df_test, config)
        self.true_numerical_columns = [
                'Census_ProcessorCoreCount',
                'Census_PrimaryDiskTotalCapacity',
                'Census_SystemVolumeTotalCapacity',
                'Census_TotalPhysicalRAM',
                'Census_InternalPrimaryDiagonalDisplaySizeInInches',
                'Census_InternalPrimaryDisplayResolutionHorizontal',
                'Census_InternalPrimaryDisplayResolutionVertical',
                'Census_InternalBatteryNumberOfCharges'
            ]
        self.binary_variables = [c for c in self.df_all.columns if self.df_all[c].nunique() == 2]
        self.categorical_columns = [c for c in self.df_all.columns
                                    if (c not in self.true_numerical_columns) & (c not in self.binary_variables)]

        self.frequency_encoded_variables = [
                                            'Census_OEMModelIdentifier',
                                            'CityIdentifier',
                                            'Census_FirmwareVersionIdentifier',
                                            'AvSigVersion',
                                            'Census_ProcessorModelIdentifier',
                                            'Census_OEMNameIdentifier',
                                            'DefaultBrowsersIdentifier'
                                            ]

    def _frequency_encoding(self, variable):
        t = self.df_all[variable].value_counts().reset_index()
        t = t.reset_index()
        t.loc[t[variable] == 1, 'level_0'] = np.nan
        t.set_index('index', inplace=True)
        max_label = t['level_0'].max() + 1
        t.fillna(max_label, inplace=True)
        return t.to_dict()['level_0']

    def frequent_encoding(self):
        for variable in tqdm(self.frequency_encoded_variables):
            freq_enc_dict = self._frequency_encoding(variable)
            self.df_all[variable] = self.df_all[variable].map(lambda x: freq_enc_dict.get(x, np.nan))
            self.categorical_columns.remove(variable)

    def category_encoding(self):
        """
        给不同类别编码, 编成0, 1, 2, 3 ...的形式, 一个类别对应一个数字
        :return:
        """
        indexer = {}
        for col in tqdm(self.categorical_columns):
            if col == 'MachineIdentifier':
                continue
            _, indexer[col] = pd.factorize(self.df_all[col])

        for col in tqdm(self.categorical_columns):
            if col == 'MachineIdentifier':
                continue
            self.df_all[col] = indexer[col].get_indexer(self.df_all[col])


def feature_engineer():
    mmp_config = config.Config()

    numerics = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numerical_columns = [c for c, v in mmp_config.DTYPES.items() if v in numerics]
    categorical_columns = [c for c, v in mmp_config.DTYPES.items() if v not in numerics]
    retained_columns = numerical_columns + categorical_columns
    df_train = pd.read_csv(mmp_config.TRAIN_PATH, nrows=mmp_config.NROWS, dtype=mmp_config.DTYPES, usecols=retained_columns)
    retained_columns.remove('HasDetections')
    df_test = pd.read_csv(mmp_config.TEST_PATH, nrows=mmp_config.NROWS, dtype=mmp_config.DTYPES, usecols=retained_columns)

    dataset = MMPDataSet(df_train, df_test, mmp_config)

    dataset.frequent_encoding()
    dataset.category_encoding()
    dataset.df_all = dataset.drop_cols(dataset.df_all, ['MachineIdentifier', 'ProductName', 'Census_DeviceFamily'])

    dataset.get_df_train().to_csv(mmp_config.TRAIN_FEATURE_PATH, index=False)
    dataset.get_df_test().to_csv(mmp_config.TEST_FEATURE_PATH, index=False)
    dataset.get_label().to_csv(mmp_config.LABEL_PATH, index=False)


if __name__ == '__main__':
    with timer('Feature Engineer'):
        feature_engineer()
#

