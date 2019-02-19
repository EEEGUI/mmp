from contextlib import contextmanager
import time
import pandas as pd
import os
from datetime import datetime
import json
import psutil


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.1f}mins".format(title, (time.time() - t0)/60))


def submission(config, pred, is_compres):
    print('Generating submission...')
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    df = pd.read_csv(config.TEST_PATH, usecols=[config.KEY], nrows=len(pred))
    df[config.LABEL_COL_NAME] = pred
    if not os.path.exists(config.OUTPUT):
        os.mkdir(config.OUTPUT)
    if is_compres:
        df.to_csv(os.path.join(config.OUTPUT, 'submission_%s.csv.zip' % now), index=False, compression='zip')
    else:
        df.to_csv(os.path.join(config.OUTPUT, 'submission_%s.csv' % now), index=False)
    # os.system('kaggle competitions submit -c microsoft-malware-prediction -f ./data/output/submission_%s.csv.zip -m "Message"' % now)


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


def save_as_json(dict_, path):
    with open(path, 'w+') as f:
        json.dump(dict_, f)


def read_json(path):
    with open(path, 'r') as f:
        dict_ = json.load(f)
        return dict_


def get_memory_state():
        phymem = psutil.virtual_memory()
        line = "Memory: %5s%% %6s/%s"%(
            phymem.percent,
            str(int(phymem.used/1024/1024))+"M",
            str(int(phymem.total/1024/1024))+"M"
            )
        return line



