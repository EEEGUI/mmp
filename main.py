from utils.config import Config
from utils.model import LGBM, XDeepfm
from utils.utils import *
import pandas as pd
import warnings
from feature_engineer import feature_engineer_sparse_matrix
import gc
import os
import sys
from utils.log import Logger
sys.stdout = Logger("log.txt", sys.stdout)

warnings.filterwarnings('ignore', category=Warning)


def train(train_feature, test_feature, label, load_data):
    config = Config
    if load_data:
        train_feature = pd.read_hdf(config.TRAIN_FEATURE_PATH, key='data')
        test_feature = pd.read_hdf(config.TEST_FEATURE_PATH, key='data')
        label = pd.read_hdf(config.LABEL_PATH, key='data')
    # config.CATEGORY_VARIABLES = [c for c in train_feature.columns if (c not in config.TRUE_NUMERICAL_COLUMNS) & (c.split('_')[0] not in ['cateasnum'])]
    config.CATEGORY_VARIABLES = 'auto'
    print('%d features are used for training...' % (train_feature.shape[1]))
    if config.MODEL == 'lgbm':
        model = LGBM(config, train_feature, label, test_feature)
    else:
        model = XDeepfm(config, train_feature, label, test_feature)
    model.k_fold_train()
    # model.train()


def main():
    mmpconfig = Config()
    for min_value, max_value in [(0.2, 0.8)]:
        """
        mmpconfig.MIN = min_value
        mmpconfig.MAX = max_value
        with timer('Feature Engineer'):
            # train_feature, test_feature, label = feature_engineer(save_feature=True)
            train_feature, test_feature, label = feature_engineer_sparse_matrix(mmpconfig, save_data=True)
        
        with timer('Training'):
            train(train_feature, test_feature, label, load_data=True)
        """
        with timer('Training'):
            train(None, None, None, load_data=True)

        gc.collect()
    # with timer('Training'):
    #     train(None, None, None, True)


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = ""
    main()
# kaggle competitions submit -c microsoft-malware-prediction -f submission.csv -m "Message"
