import pandas as pd
from utils import config
from utils import dataset
from utils.utils import timer
import numpy as np
from tqdm import tqdm


class MMPDataSet(dataset.DataSet):
    def __init__(self, df_train, df_test, config):
        super(MMPDataSet, self).__init__(df_train, df_test, config)
        self.true_numerical_variables = config.TRUE_NUMERICAL_COLUMNS
        self.frequency_encoded_variables = config.FREQUENT_ENCODED_COLUMNS
        self.label_encoded_variables = [c for c in self.df_all.columns
                                        if (c not in self.true_numerical_variables) &
                                        (c not in self.frequency_encoded_variables) &
                                        (c != self.config.KEY)]

    def _frequency_encoding(self, variable):
        t = self.df_all[variable].value_counts().reset_index()
        t = t.reset_index()
        t.loc[t[variable] == 1, 'level_0'] = np.nan
        t.set_index('index', inplace=True)
        max_label = t['level_0'].max() + 1
        t.fillna(max_label, inplace=True)
        return t.to_dict()['level_0']

    def frequent_encoding(self, cols_to_encode):
        for variable in tqdm(cols_to_encode):
            freq_enc_dict = self._frequency_encoding(variable)
            self.df_all[variable] = self.df_all[variable].map(lambda x: freq_enc_dict.get(x, np.nan))

    def category_encoding(self):
        """
        给不同类别编码, 编成0, 1, 2, 3 ...的形式, 一个类别对应一个数字
        :return:
        """
        self.frequent_encoding(self.frequency_encoded_variables)
        self.label_encoding(self.label_encoded_variables)

    def drop_key(self):
        self.df_all = self.drop_cols(self.df_all, [self.config.KEY])


def feature_engineer():
    mmp_config = config.Config()

    numerics = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numerical_columns = [c for c, v in mmp_config.DTYPES.items() if v in numerics]
    categorical_columns = [c for c, v in mmp_config.DTYPES.items() if v not in numerics]
    retained_columns = numerical_columns + categorical_columns
    print('Reading train.csv...')
    df_train = pd.read_csv(mmp_config.TRAIN_PATH,
                           nrows=mmp_config.NROWS,
                           dtype=mmp_config.DTYPES,
                           usecols=retained_columns)
    retained_columns.remove('HasDetections')
    print('Reading test.csv...')
    df_test = pd.read_csv(mmp_config.TEST_PATH,
                          nrows=mmp_config.NROWS,
                          dtype=mmp_config.DTYPES,
                          usecols=retained_columns)

    dataset = MMPDataSet(df_train, df_test, mmp_config)

    del df_train
    del df_test

    print('Label encoding...')
    dataset.category_encoding()
    dataset.drop_key()

    # dataset.get_df_train().to_csv(mmp_config.TRAIN_FEATURE_PATH, index=False)
    # dataset.get_df_test().to_csv(mmp_config.TEST_FEATURE_PATH, index=False)
    # dataset.get_label().to_csv(mmp_config.LABEL_PATH, index=False)

    return dataset.get_df_train(), dataset.get_df_test(), dataset.get_label()


if __name__ == '__main__':
    with timer('Feature Engineer'):
        feature_engineer()

