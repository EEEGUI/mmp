from utils import config
from utils import dataset
from utils.utils import *
import numpy as np
from tqdm import tqdm
import pandas_profiling as pdf
import warnings
from utils.feature_selector import FeatureSelector
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


warnings.filterwarnings('ignore')


class MMPDataSet(dataset.DataSet):
    def __init__(self, df_train, df_test, config):
        super(MMPDataSet, self).__init__(df_train, df_test, config)

        self.true_numerical_variables = config.TRUE_NUMERICAL_COLUMNS   # 具有实际意义的数值型字段
        self.date_dict = np.load(config.VERSION_TIME_DICT_PATH)[()]

        self.df_all['AvSigVersion'] = self.df_all['AvSigVersion'].map(self.date_dict)
        self.df_all.iloc[:self.len_train, :] = self.df_all.iloc[:self.len_train, :].sort_values(by='AvSigVersion',
                                                                                                ascending=False)

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
        self.true_numerical_variables = list(set(self.true_numerical_variables) - set(cols_to_drop))

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

    def feature_alignment(self):
        """
        将样本数目少的类别归为一类
        对齐训练集和测试集特征各类别的频率分布，将分布差别大的类别归为一类
        :return:
        """
        for usecol in tqdm(self.df_all.columns.tolist()[1:]):
            self.df_all[usecol] = self.df_all[usecol].astype('str')

            # Fit LabelEncoder
            le = LabelEncoder().fit(
                np.unique(self.df_all[usecol].unique().tolist()))

            # At the end 0 will be used for dropped values
            self.df_all[usecol] = le.transform(self.df_all[usecol]) + 1

            agg_tr = (self.get_df_train()
                      .groupby([usecol])
                      .aggregate({'MachineIdentifier': 'count'})
                      .reset_index()
                      .rename({'MachineIdentifier': 'Train'}, axis=1))
            agg_te = (self.get_df_test()
                      .groupby([usecol])
                      .aggregate({'MachineIdentifier': 'count'})
                      .reset_index()
                      .rename({'MachineIdentifier': 'Test'}, axis=1))

            agg = pd.merge(agg_tr, agg_te, on=usecol, how='outer').replace(np.nan, 0)
            # Select values with more than 1000 observations
            agg = agg[(agg['Train'] > 1000)].reset_index(drop=True)
            agg['Total'] = agg['Train'] + agg['Test']
            # Drop unbalanced values
            agg = agg[(agg['Train'] / agg['Total'] > 0.2) & (agg['Train'] / agg['Total'] < 0.8)]
            agg[usecol + 'Copy'] = agg[usecol]

            self.df_all[usecol] = (pd.merge(self.df_all[[usecol]],
                                      agg[[usecol, usecol + 'Copy']],
                                      on=usecol, how='left')[usecol + 'Copy']
                             .replace(np.nan, 0).astype('int').astype('category'))
            del le, agg_tr, agg_te, agg, usecol
        self.drop_key()

    def one_hot_encoding(self):
        ohe = OneHotEncoder(categories='auto', sparse=True, dtype='uint8').fit(self.df_all)
        self.df_all = ohe.transform(self.df_all)

    def generate_feature(self):
        """
        生成新特征
        :return:
        """
        # Week
        first = datetime(2018, 1, 1)
        datedict2 = {}
        for x in self.date_dict: datedict2[x] = (self.date_dict[x] - first).days // 7
        self.df_all['Week'] = self.df_all['AvSigVersion'].map(datedict2)

        self.df_all['EngineVersion_2'] = self.df_all['EngineVersion'].apply(lambda x: x.split('.')[2]).astype(
            'category')
        self.df_all['EngineVersion_3'] = self.df_all['EngineVersion'].apply(lambda x: x.split('.')[3]).astype(
            'category')

        self.df_all['AppVersion_1'] = self.df_all['AppVersion'].apply(lambda x: x.split('.')[1]).astype('category')
        self.df_all['AppVersion_2'] = self.df_all['AppVersion'].apply(lambda x: x.split('.')[2]).astype('category')
        self.df_all['AppVersion_3'] = self.df_all['AppVersion'].apply(lambda x: x.split('.')[3]).astype('category')

        self.df_all['AvSigVersion_0'] = self.df_all['AvSigVersion'].apply(lambda x: x.split('.')[0]).astype('category')
        self.df_all['AvSigVersion_1'] = self.df_all['AvSigVersion'].apply(lambda x: x.split('.')[1]).astype('category')
        self.df_all['AvSigVersion_2'] = self.df_all['AvSigVersion'].apply(lambda x: x.split('.')[2]).astype('category')

        # self.df_all['OsBuildLab_0'] = self.df_all['OsBuildLab'].astype('str').apply(lambda x: x.split('.')[0]).astype('category')
        # self.df_all['OsBuildLab_1'] = self.df_all['OsBuildLab'].astype('str').apply(lambda x: x.split('.')[1]).astype('category')
        # self.df_all['OsBuildLab_2'] = self.df_all['OsBuildLab'].astype('str').apply(lambda x: x.split('.')[2]).astype('category')
        # self.df_all['OsBuildLab_3'] = self.df_all['OsBuildLab'].astype('str').apply(lambda x: x.split('.')[3]).astype('category')
        # self.df_all['OsBuildLab_40'] = self.df_all['OsBuildLab'].apply(lambda x: x.split('.')[4].split('-')[0]).astype('category')
        # self.df_all['OsBuildLab_41'] = self.df_all['OsBuildLab'].apply(lambda x: x.split('.')[4].split('-')[1]).astype('category')

        self.df_all['Census_OSVersion_0'] = self.df_all['Census_OSVersion'].apply(lambda x: x.split('.')[0]).astype(
            'category')
        self.df_all['Census_OSVersion_1'] = self.df_all['Census_OSVersion'].apply(lambda x: x.split('.')[1]).astype(
            'category')
        self.df_all['Census_OSVersion_2'] = self.df_all['Census_OSVersion'].apply(lambda x: x.split('.')[2]).astype(
            'category')
        self.df_all['Census_OSVersion_3'] = self.df_all['Census_OSVersion'].apply(lambda x: x.split('.')[3]).astype(
            'category')

        # https://www.kaggle.com/adityaecdrid/simple-feature-engineering-xd
        self.df_all['primary_drive_c_ratio'] = self.df_all['Census_SystemVolumeTotalCapacity'] / self.df_all[
            'Census_PrimaryDiskTotalCapacity']
        self.df_all['non_primary_drive_MB'] = self.df_all['Census_PrimaryDiskTotalCapacity'] - self.df_all[
            'Census_SystemVolumeTotalCapacity']

        self.df_all['aspect_ratio'] = self.df_all['Census_InternalPrimaryDisplayResolutionHorizontal'] / self.df_all[
            'Census_InternalPrimaryDisplayResolutionVertical']

        self.df_all['monitor_dims'] = self.df_all['Census_InternalPrimaryDisplayResolutionHorizontal'].astype(
            str) + '*' + self.df_all[
                                          'Census_InternalPrimaryDisplayResolutionVertical'].astype('str')
        self.df_all['monitor_dims'] = self.df_all['monitor_dims'].astype('category')

        self.df_all['dpi'] = ((self.df_all['Census_InternalPrimaryDisplayResolutionHorizontal'] ** 2 + self.df_all[
            'Census_InternalPrimaryDisplayResolutionVertical'] ** 2) ** .5) / (
                                 self.df_all['Census_InternalPrimaryDiagonalDisplaySizeInInches'])

        self.df_all['dpi_square'] = self.df_all['dpi'] ** 2

        self.df_all['MegaPixels'] = (self.df_all['Census_InternalPrimaryDisplayResolutionHorizontal'] * self.df_all[
            'Census_InternalPrimaryDisplayResolutionVertical']) / 1e6

        self.df_all['Screen_Area'] = (self.df_all['aspect_ratio'] * (
                    self.df_all['Census_InternalPrimaryDiagonalDisplaySizeInInches'] ** 2)) / (
                                             self.df_all['aspect_ratio'] ** 2 + 1)

        self.df_all['ram_per_processor'] = self.df_all['Census_TotalPhysicalRAM'] / self.df_all[
            'Census_ProcessorCoreCount']

        self.df_all['new_num_0'] = self.df_all['Census_InternalPrimaryDiagonalDisplaySizeInInches'] / self.df_all[
            'Census_ProcessorCoreCount']

        self.df_all['new_num_1'] = self.df_all['Census_ProcessorCoreCount'] * self.df_all[
            'Census_InternalPrimaryDiagonalDisplaySizeInInches']

        self.df_all['Census_IsFlightingInternal'] = self.df_all['Census_IsFlightingInternal'].fillna(1)
        self.df_all['Census_ThresholdOptIn'] = self.df_all['Census_ThresholdOptIn'].fillna(1)
        self.df_all['Census_IsWIMBootEnabled'] = self.df_all['Census_IsWIMBootEnabled'].fillna(1)
        self.df_all['Wdft_IsGamer'] = self.df_all['Wdft_IsGamer'].fillna(0)


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

    print('Drop features')
    dataset.drop_key()
    # dataset.drop_features()

    # print('Generate new feature')
    # dataset.category_to_frequent()

    # print('Label encoding...')
    # dataset.category_encoding()

    print('Category to number...')
    dataset.category_encoding()
    dataset.category_to_number()

    print('%d features are used in train' % dataset.df_all.shape[1])
    print('The length of train is %d' % df_train_length)
    print('The length of test is %d' % df_test_length)

    dataset.reduce_memory_usage()

    if save_feature:
        dataset.get_df_train().to_hdf(mmp_config.TRAIN_FEATURE_PATH, key='data', format='t')
        dataset.get_df_test().to_hdf(mmp_config.TEST_FEATURE_PATH, key='data', format='t')
        dataset.get_label().to_hdf(mmp_config.LABEL_PATH, key='data', format='t')

    return dataset.get_df_train(), dataset.get_df_test(), dataset.get_label()


def feature_engineer_sparse_matrix():
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

    dataset.generate_feature()
    dataset.feature_alignment()
    dataset.one_hot_encoding()

    print('%d features are used in train' % dataset.df_all.shape[1])
    print('The length of train is %d' % df_train_length)
    print('The length of test is %d' % df_test_length)

    # dataset.reduce_memory_usage()

    return dataset.df_all[:dataset.len_train], dataset.df_all[dataset.len_train:], dataset.get_label()


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
    with timer('Feature Engineering '):
        # feature_engineer(save_feature=True)
        feature_engineer_sparse_matrix()