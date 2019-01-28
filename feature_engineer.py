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
        # self.true_numerical_variables = config.TRUE_NUMERICAL_COLUMNS
        # self.frequency_encoded_variables = config.FREQUENT_ENCODED_COLUMNS
        # self.label_encoded_variables = [c for c in self.df_all.columns
        #                                 if (c not in self.true_numerical_variables) &
        #                                 (c not in self.frequency_encoded_variables) &
        #                                 (c != self.config.KEY)]
        # self.category_variables = self.frequency_encoded_variables + self.label_encoded_variables

        self.ori_number_variables = [v for v in config.DTYPES if v in config.NUMBER_TYPE]   # 原始数据即为数值型
        self.ori_category_variables = [v for v in config.DTYPES if v not in config.NUMBER_TYPE]  # 原始数据不为数值型
        self.label_encoded_variables = self.ori_category_variables

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
        # self.frequent_encoding(self.frequency_encoded_variables)
        self.label_encoding(self.label_encoded_variables)

    def cal_category_frequency(self, variable):
        t = self.df_all[variable].value_counts().reset_index()
        t = t.reset_index()
        t.set_index('index', inplace=True)
        t['level_0'] = t['level_0'] / len(t)
        t['level_0'] = t['level_0'].astype('float16')
        return t.to_dict()['level_0']

    def category_to_frequent(self):
        for variable in tqdm(self.category_variables):
            category_2_frequency_dict = self.cal_category_frequency(variable)
            self.df_all[variable + '_f'] = self.df_all[variable].map(lambda x: category_2_frequency_dict.get(x, np.nan))

    def drop_key(self):
        self.df_all = self.drop_cols(self.df_all, [self.config.KEY])
        self.update_features([self.config.KEY])

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
        self.update_features(cols_to_drop)

    def update_features(self, cols_to_drop):
        """
        删除部分特征后，维护当前最新的类别特征和数值型特征，防止出现key error
        :return:
        """
        self.frequency_encoded_variables = list(set(self.frequency_encoded_variables) - set(cols_to_drop))
        self.label_encoded_variables = list(set(self.label_encoded_variables) - set(cols_to_drop))
        self.category_variables = list(set(self.category_variables) - set(cols_to_drop))

    def find_useless_feature(self):
        fs = FeatureSelector(data=self.get_df_train(), labels=self.get_label())
        fs.identify_all(selection_params={'missing_threshold': 0.6, 'correlation_threshold': 0.98,
                                          'task': 'classification', 'eval_metric': 'auc',
                                          'cumulative_importance': 0.99})
        df_train_drop_variables = fs.remove(methods='all', keep_one_hot=True)
        remain_columns = df_train_drop_variables.columns
        self.df_all = self.df_all.loc[:, remain_columns]

        save_as_json(fs.ops, self.config.FEATURE_TO_DROP_JSON)

    def reduce_memory_usage(self):
        verbose = True
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        start_mem = self.df_all.memory_usage().sum() / 1024 ** 2
        for col in tqdm(self.df_all.columns):
            col_type = self.df_all[col].dtypes
            if col_type in numerics:
                c_min = self.df_all[col].min()
                c_max = self.df_all[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        self.df_all[col] = self.df_all[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        self.df_all[col] = self.df_all[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        self.df_all[col] = self.df_all[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        self.df_all[col] = self.df_all[col].astype(np.int64)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        self.df_all[col] = self.df_all[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        self.df_all[col] = self.df_all[col].astype(np.float32)
                    else:
                        self.df_all[col] = self.df_all[col].astype(np.float64)
        end_mem = self.df_all.memory_usage().sum() / 1024 ** 2
        if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (
                    start_mem - end_mem) / start_mem))


def feature_engineer(save_feature=True):
    mmp_config = config.Config()
    print('Reading train.h5...')
    df_train = pd.read_hdf(mmp_config.TRAIN_H5_PATH, key='data')
    if mmp_config.RANDOM_SAMPLE_PERCENTAGE:
        df_train = df_train.sample(frac=mmp_config.RANDOM_SAMPLE_PERCENTAGE, random_state=mmp_config.RANDOM_STATE)

    df_train_length = len(df_train)
    print('Reading test.h5...')
    df_test = pd.read_hdf(mmp_config.TEST_H5_PATH, key='data')
    df_test_length = len(df_test)

    dataset = MMPDataSet(df_train, df_test, mmp_config)

    del df_train
    del df_test

    # print('Split feature...')
    # dataset.split_feature()

    # print('Drop features')
    # dataset.drop_key()
    # dataset.drop_features()

    # print('Generate new feature')
    # dataset.category_to_frequent()

    print('Label encoding...')
    dataset.category_encoding()

    print('%d features are used in train' % dataset.df_all.shape[1])
    print('The length of train is %d' % df_train_length)
    print('The length of test is %d' % df_test_length)

    dataset.reduce_memory_usage()

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
    with timer('Save to train.h5'):
        save_as_h5(df_train, mmp_config.TRAIN_H5_PATH)

    del df_train

    print('Reading test.csv...')
    with timer('Read test.csv'):
        df_test = pd.read_csv(mmp_config.TEST_PATH,
                              nrows=mmp_config.NROWS,
                              dtype=mmp_config.DTYPES)

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
    feature_engineer(save_feature=True)