from utils.config import Config
from utils.model import LGBM
from utils.utils import *
import pandas as pd
import warnings
from feature_engineer import feature_engineer

warnings.filterwarnings('ignore', category=Warning)


def train(train_feature, test_feature, label):
    config = Config

    # train_feature = pd.read_csv(config.TRAIN_FEATURE_PATH)
    # test_feature = pd.read_csv(config.TEST_FEATURE_PATH)
    # label = pd.read_csv(config.LABEL_PATH)
    config.CATEGORY_VARIABLES = [c for c in train_feature.columns if c not in config.TRUE_NUMERICAL_COLUMNS]

    lgbm = LGBM(config, train_feature, label, test_feature)
    submission(config, lgbm.train())


if __name__ == '__main__':
    with timer('Feature Engineer'):
        train_feature, test_feature, label = feature_engineer()
    with timer('Training'):
        train(train_feature, test_feature, label)
