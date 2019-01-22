from contextlib import contextmanager
import time
import pandas as pd
import os
from utils.config import Config

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.4f}mins".format(title, (time.time() - t0)))


def submission(config, pred):
    df = pd.read_csv(config.TEST_PATH, usecols=[config.KEY], nrows=config.NROWS)
    df[config.LABEL_COL_NAME] = pred
    if not os.path.exists(config.OUTPUT):
        os.mkdir(config.OUTPUT)
    df.to_csv(os.path.join(config.OUTPUT, 'submission.csv.zip'), index=False, compression='zip')


def drop_cols(df, list_cols):
    """
    del list_cols from df
    :param df:
    :param list_cols: list, cols to delete
    :return: df without list_cols
    """
    return df.drop(labels=list_cols, axis=1)


def save_as_h5(df, path):
    h5 = pd.HDFStore(path, 'w')
    h5['data'] = df
    h5.close()


def convert_format():
    mmp_config = Config()

    numerics = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numerical_columns = [c for c, v in mmp_config.DTYPES.items() if v in numerics]
    categorical_columns = [c for c, v in mmp_config.DTYPES.items() if v not in numerics]
    retained_columns = numerical_columns + categorical_columns
    print('Reading train.csv...')
    with timer('Read train.csv'):
        df_train = pd.read_csv(mmp_config.TRAIN_PATH,
                               nrows=mmp_config.NROWS,
                               dtype=mmp_config.DTYPES,
                               usecols=retained_columns)
    retained_columns.remove('HasDetections')
    print('Reading test.csv...')
    with timer('Read test.csv'):
        df_test = pd.read_csv(mmp_config.TEST_PATH,
                              nrows=mmp_config.NROWS,
                              dtype=mmp_config.DTYPES,
                              usecols=retained_columns)

    with timer('Save to train.h5'):
        save_as_h5(df_train, mmp_config.TRAIN_H5_PATH)
    with timer('Save to test.h5'):
        save_as_h5(df_test, mmp_config.TEST_H5_PATH)

    with timer('Read train.h5'):
        df_train = pd.read_hdf(mmp_config.TRAIN_H5_PATH, key='data')

    with timer('Read test.h5'):
        df_test = pd.read_hdf(mmp_config.TEST_H5_PATH, key='data')


if __name__ == '__main__':
    convert_format()



