from contextlib import contextmanager
import time
import pandas as pd
import os


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.1f}mins".format(title, (time.time() - t0)/60))


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
    df.to_hdf(path, key='data', format='table')



