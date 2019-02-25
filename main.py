from utils.config import Config
from utils.model import LGBM
from utils.utils import *
import pandas as pd
import warnings
from feature_engineer import feature_engineer_sparse_matrix

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
    lgbm = LGBM(config, train_feature, label, test_feature)
    submission(config, lgbm.k_fold_train(), True)


def main():
    with timer('Feature Engineer'):
        # train_feature, test_feature, label = feature_engineer(save_feature=True)
        train_feature, test_feature, label = feature_engineer_sparse_matrix()
    with timer('Training'):
        train(train_feature, test_feature, label, load_data=False)
    # with timer('Training'):
    #     train(None, None, None, True)


if __name__ == '__main__':
    main()

# kaggle competitions submit -c microsoft-malware-prediction -f submission.csv -m "Message"
