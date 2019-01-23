from contextlib import contextmanager
import time
import pandas as pd
import os
from datetime import datetime


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.1f}mins".format(title, (time.time() - t0)/60))


def submission(config, pred, is_compres):
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    df = pd.read_csv(config.TEST_PATH, usecols=[config.KEY], nrows=config.NROWS)
    df[config.LABEL_COL_NAME] = pred
    if not os.path.exists(config.OUTPUT):
        os.mkdir(config.OUTPUT)
    if is_compres:
        df.to_csv(os.path.join(config.OUTPUT, 'submission_%s.csv.zip' % now), index=False, compression='zip')
    else:
        df.to_csv(os.path.join(config.OUTPUT, 'submission_%s.csv' % now), index=False)


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



