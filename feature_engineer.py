from utils import config
from utils import dataset
from utils.utils import *
import numpy as np
from tqdm import tqdm
import pandas_profiling as pdf
import warnings
from utils.feature_selector import FeatureSelector


warnings.filterwarnings('ignore')


class MMPDataSet(dataset.DataSet):
    def __init__(self, df_train, df_test, config):
        super(MMPDataSet, self).__init__(df_train, df_test, config)
        self.true_numerical_variables = config.TRUE_NUMERICAL_COLUMNS
        self.frequency_encoded_variables = config.FREQUENT_ENCODED_COLUMNS
        self.label_encoded_variables = [c for c in self.df_all.columns
                                        if (c not in self.true_numerical_variables) &
                                        (c not in self.frequency_encoded_variables) &
                                        (c != self.config.KEY)]
        self.category_variables = self.frequency_encoded_variables + self.label_encoded_variables

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

    def category_frequent(self):
        for variable in tqdm(self.category_variables):
            self.df_all[variable+'_f'] = self.df_all.groupby(variable)[variable].transform('count') / len(self.df_all)

    def drop_key(self):
        self.df_all = self.drop_cols(self.df_all, [self.config.KEY])

    def split_feature(self):
        for variable in tqdm(self.config.COLUMNS_TO_SPLIT):
            split_result = self.df_all[variable].apply(lambda x: x.split('.'))
            for i in range(len(split_result[0])):
                self.df_all['%s_%d' % (variable, i)] = split_result.apply(lambda x: self._to_int(x[i]))

    @staticmethod
    def _to_int(x):
        if set(x) < set(range(0, 10)):
            return int(x)
        else:
            return np.nan

    def drop_features(self):
        cols_to_drop_dict = read_json(self.config.FEATURE_TO_DROP_JSON)
        cols_to_drop = []
        for key in cols_to_drop_dict:
            cols_to_drop += cols_to_drop_dict[key]
        cols_to_drop += self.config.COLUMNS_TO_DROP
        cols_to_drop = list(set(cols_to_drop))
        self.df_all = self.drop_cols(self.df_all, cols_to_drop)

    def find_useless_feature(self):
        fs = FeatureSelector(data=self.get_df_train(), labels=self.get_label())
        fs.identify_all(selection_params={'missing_threshold': 0.6, 'correlation_threshold': 0.98,
                                          'task': 'classification', 'eval_metric': 'auc',
                                          'cumulative_importance': 0.99})
        df_train_drop_variables = fs.remove(methods='all', keep_one_hot=True)
        remain_columns = df_train_drop_variables.columns
        self.df_all = self.df_all.loc[:, remain_columns]

        save_as_json(fs.ops, self.config.FEATURE_TO_DROP_JSON)


def feature_engineer(save_feature=True):
    mmp_config = config.Config()
    print('Reading train.h5...')
    df_train = pd.read_hdf(mmp_config.TRAIN_H5_PATH, key='data')

    print('Reading test.h5...')
    df_test = pd.read_hdf(mmp_config.TEST_H5_PATH, key='data')

    dataset = MMPDataSet(df_train, df_test, mmp_config)

    del df_train
    del df_test

    # print('Split feature...')
    # dataset.split_feature()

    print('Label encoding...')
    dataset.category_encoding()

    print('Generate new feature')
    dataset.category_frequent()

    print('Drop some feature...')
    dataset.drop_key()

    if save_feature:
        dataset.get_df_train().to_hdf(mmp_config.TRAIN_FEATURE_PATH, key='data', format='t')
        dataset.get_df_test().to_hdf(mmp_config.TEST_FEATURE_PATH, key='data', format='t')
        dataset.get_label().to_hdf(mmp_config.LABEL_PATH, key='data', format='t')

    return dataset.get_df_train(), dataset.get_df_test(), dataset.get_label()


def convert_format():
    mmp_config = config.Config()
    print('Reading train.csv...')
    with timer('Read train.csv'):
        df_train = pd.read_csv(mmp_config.TRAIN_PATH,
                               nrows=mmp_config.NROWS,
                               dtype=mmp_config.DTYPES)

    print('Reading test.csv...')
    with timer('Read test.csv'):
        df_test = pd.read_csv(mmp_config.TEST_PATH,
                              nrows=mmp_config.NROWS,
                              dtype=mmp_config.DTYPES)

    with timer('Save to train.h5'):
        save_as_h5(df_train, mmp_config.TRAIN_H5_PATH)
    with timer('Save to test.h5'):
        save_as_h5(df_test, mmp_config.TEST_H5_PATH)


def feature_report():
    mmp_config = config.Config()
    print('Reading train.h5...')
    with timer('Reading train.h5'):
        df_train = pd.read_hdf(mmp_config.TRAIN_H5_PATH, key='data')

    print('Reading test.h5...')
    with timer('Reading test.h5'):
        df_test = pd.read_hdf(mmp_config.TEST_H5_PATH, key='data')

    with timer('Train report'):
        train_report = pdf.ProfileReport(df_train)
        train_report.to_file(mmp_config.TRAIN_REPORT_PATH)

    with timer('Test report'):
        test_report = pdf.ProfileReport(df_test)
        test_report.to_file(mmp_config.TEST_REPORT_PATH)


if __name__ == '__main__':
    # convert_format()
    feature_engineer(save_feature=False)