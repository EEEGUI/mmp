from contextlib import contextmanager
import time
import pandas as pd
import os


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}mins".format(title, (time.time() - t0)/60))


def submission(config, pred):
    df = pd.read_csv(config.TEST_PATH, usecols=[config.KEY], nrows=config.NROWS)
    df[config.LABEL_COL_NAME] = pred
    if not os.path.exists(config.OUTPUT):
        os.mkdir(config.OUTPUT)
    df.to_csv(os.path.join(config.OUTPUT, 'submission.csv'), index=False)


