import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import rankdata
from utils.utils import *
from scipy.sparse import csr_matrix
import tensorflow as tf
import ctrNet
import gc
from src import misc_utils as utils


class LGBM:
    def __init__(self, config, train_features, train_labels, test_features):
        self.train_features = train_features
        self.train_labels = train_labels
        self.test_features = test_features
        self.config = config
        self.feature_importance = None
        self.use_sparse_matrix = False if isinstance(train_features, pd.DataFrame) else True

    def k_fold_train(self, **kwargs):
        if self.config.KFOLD:
            k_fold = KFold(n_splits=self.config.N_FOLDS, shuffle=True, random_state=712)
        else:
            k_fold = StratifiedKFold(n_splits=self.config.N_FOLDS, shuffle=True, random_state=712)
        valid_scores = []
        # train_scores = []
        feature_importance_values = np.zeros(self.train_features.shape[1])
        test_predictions = np.zeros(self.test_features.shape[0])

        for train_indices, valid_indices in k_fold.split(self.train_features, None if self.config.KFOLD else self.train_labels):

            if self.use_sparse_matrix:
                train_x, train_y = self.train_features[train_indices], self.train_labels.iloc[train_indices, :]
                valid_x, valid_y = self.train_features[valid_indices], self.train_labels.iloc[valid_indices, :]
            else:
                train_x, train_y = self.train_features.iloc[train_indices, :], self.train_labels.iloc[train_indices, :]
                valid_x, valid_y = self.train_features.iloc[valid_indices, :], self.train_labels.iloc[valid_indices, :]

            if self.use_sparse_matrix:
                train_x, valid_x, self.test_features = csr_matrix(train_x, dtype='float32'), \
                                                       csr_matrix(valid_x, dtype='float32'), \
                                                       csr_matrix(self.test_features, dtype='float32')

            lgb_train = lgb.Dataset(train_x, train_y)
            lgb_eval = lgb.Dataset(valid_x, valid_y)
            gbm = lgb.train(self.config.PARAM, lgb_train,
                            valid_sets=[lgb_eval],
                            categorical_feature=self.config.CATEGORY_VARIABLES)
            feature_importance_values += gbm.feature_importance() / k_fold.n_splits
            valid_scores.append(gbm.best_score['valid_0']['auc'])
            # valid_scores.append(gbm.best_score['valid_1']['auc'])
            # train_scores.append(gbm.best_score['training']['auc'])
            # test_predictions += gbm.predict(self.test_features, num_iteration=gbm.best_iteration) / k_fold.n_splits
            test_predictions = np.add(test_predictions,
                                      rankdata(gbm.predict(self.test_features, num_iteration=gbm.best_iteration))
                                      / test_predictions.shape[0])
        test_predictions /= k_fold.n_splits
        # print('Average training`s AUC is %.5f, std=%.5f' % (np.mean(train_scores), np.std(train_scores)))
        print('Average valid`s AUC is %.5f, std=%.5f' % (np.mean(valid_scores), np.std(valid_scores)))
        print('Saving model...')
        # save model to file
        gbm.save_model(self.config.MODEL_SAVING_PATH)
        if not self.use_sparse_matrix:
            self.feature_importance = pd.DataFrame({'feature': self.train_features.columns,
                                                'importance': gbm.feature_importance()})
            self.plot_feature_importance()
            print('Saving model...')
        del gbm

        submission(self.config, test_predictions, True, '%.5f' % np.mean(valid_scores))


    def train(self):
        train_x, valid_x, train_y, valid_y = train_test_split(self.train_features, self.train_labels, test_size=0.2,
                                                              shuffle=False, random_state=712)
        if self.use_sparse_matrix:
            train_x, valid_x, self.test_features = csr_matrix(train_x, dtype='float32'), \
                                                   csr_matrix(valid_x, dtype='float32'), \
                                                   csr_matrix(self.test_features, dtype='float32')

        lgb_train = lgb.Dataset(train_x, train_y)
        lgb_eval = lgb.Dataset(valid_x, valid_y)
        param = read_json(self.config.LIGHTGBM_BEST_PARAM)
        param = self.config.PARAM
        gbm = lgb.train(param, lgb_train,
                        valid_sets=[lgb_eval],
                        categorical_feature=self.config.CATEGORY_VARIABLES)
        print('Predicting...')
        test_predictions = gbm.predict(self.test_features, num_iteration=gbm.best_iteration)
        # save model to file
        gbm.save_model(self.config.MODEL_SAVING_PATH)

        if not self.use_sparse_matrix:
            self.feature_importance = pd.DataFrame({'feature': self.train_features.columns,
                                                'importance': gbm.feature_importance()})
            self.plot_feature_importance()
            print('Saving model...')
        print_tree = False
        if print_tree:
            print('Plotting 1th tree with graphviz...')
            graph = lgb.create_tree_digraph(gbm, tree_index=0, name='Tree1')
            graph.render(filename='assets/tree_graph')
        submission(self.config, test_predictions, True, '%.5f' % gbm.best_score['valid_0']['auc'])

        return test_predictions

    def plot_feature_importance(self):
        plt.figure(figsize=(14, 25))
        sns.barplot(x="importance",
                    y="feature",
                    data=self.feature_importance.sort_values(by="importance", ascending=False))
        plt.title('LightGBM Features (avg over folds)')
        plt.tight_layout()
        plt.savefig(self.config.FEATURE_IMPORTANCE_FIG)


class XDeepfm:
    def __init__(self, config, train_features, train_labels, test_features):
        self.train_features = train_features
        self.train_labels = train_labels
        self.test_features = test_features
        self.config = config
        self.feature_importance = None
        self.use_sparse_matrix = False if isinstance(train_features, pd.DataFrame) else True
        self.config.FM_PARAM.update({'feature_nums': self.train_features.shape[1]})
        self.param = tf.contrib.training.HParams(**self.config.FM_PARAM)

    def k_fold_train(self):
        if self.config.KFOLD:
            k_fold = KFold(n_splits=self.config.N_FOLDS, shuffle=True, random_state=712)
        else:
            k_fold = StratifiedKFold(n_splits=self.config.N_FOLDS, shuffle=True, random_state=712)
        test_predictions = np.zeros(self.test_features.shape[0])
        for train_indices, valid_indices in k_fold.split(self.train_features,
                                                         None if self.config.KFOLD else self.train_labels):
            train_x, train_y = self.train_features.iloc[train_indices, :], self.train_labels.iloc[train_indices, :]
            valid_x, valid_y = self.train_features.iloc[valid_indices, :], self.train_labels.iloc[valid_indices, :]

            model = ctrNet.build_model(self.param)
            model.train(train_data=(train_x, train_y),
                        dev_data=(valid_x, valid_y))
            print("Training Done! Inference...")

            test_predictions = np.add(test_predictions,
                                      rankdata(model.infer(dev_data=(self.test_features, test_predictions)))
                                      / test_predictions.shape[0])
            del model
            gc.collect()

        submission(self.config, test_predictions, True, '')