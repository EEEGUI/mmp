from utils.config import Config
from utils.model import LGBM
from utils.utils import *
import pandas as pd


def main():
    config = Config

    train_feature = pd.read_csv(config.TRAIN_FEATURE_PATH)
    test_feature = pd.read_csv(config.TEST_FEATURE_PATH)
    label = pd.read_csv(config.LABEL_PATH)

    train_feature = drop_cols(train_feature, ['PuaMode'])
    test_feature = drop_cols(test_feature, ['PuaMode'])

    lgbm = LGBM(config, train_feature, label, test_feature)
    submission(config, lgbm.train())


if __name__ == '__main__':
    with timer('Training'):
        main()
