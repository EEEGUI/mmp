from tqdm import tqdm
import lightgbm as lgb
import random
import itertools
import numpy as np
import warnings
from utils.config import Config
from utils.utils import *


warnings.filterwarnings('ignore')


class ParamSearch:
    def __init__(self, train_array, train_label_array, config):
        self.train_set = lgb.Dataset(train_array, train_label_array)
        self.param_grid = config.PARAM_GRID
        self.max_evals = config.SEARCH_TIME
        self.config = config

    def objective(self, hyperparameters, iteration):
        """Objective function for grid and random search. Returns
           the cross validation score from a set of hyperparameters."""

        # Number of estimators will be found using early stopping
        if 'n_estimators' in hyperparameters.keys():
            del hyperparameters['n_estimators']

            # Perform n_folds cross validation
        hyperparameters['verbose'] = -1
        hyperparameters['objective'] = 'binary'
        cv_results = lgb.cv(hyperparameters, self.train_set, nfold=3, num_boost_round=10000,
                            early_stopping_rounds=100, metrics='auc', shuffle=True)

        # results to return
        score = cv_results['auc-mean'][-1]
        estimators = len(cv_results['auc-mean'])
        hyperparameters['n_estimators'] = estimators

        return [score, hyperparameters, iteration]

    def random_search(self):
        """Random search for hyperparameter optimization"""
        best_score = 0
        best_param = {}
        if os.path.exists(self.config.LIGHTGBM_BEST_PARAM):
            best_param = read_json(self.config.LIGHTGBM_BEST_PARAM)
            if 'score' in best_param.keys():
                best_score = best_param['score']

        # Keep searching until reach max evaluations
        for i in tqdm(range(self.max_evals)):
            # Choose random hyperparameters
            hyperparameters = {k: random.sample(v, 1)[0] for k, v in self.param_grid.items()}
            hyperparameters['bagging_fraction'] = 1.0 if hyperparameters['boosting_type'] == 'goss' else hyperparameters[
                'bagging_fraction']

            # Evaluate randomly selected hyperparameters
            eval_results = self.objective(hyperparameters, i)
            print('score:%.5f:' % eval_results[0])
            if eval_results[0] > best_score:
                best_param = eval_results[1]
                best_score = eval_results[0]
                best_param['score'] = best_score
                save_as_json(best_param, config.LIGHTGBM_BEST_PARAM)

        print(best_param)
        print(best_score)

    def grid_search(self):
        """Grid search algorithm (with limit on max evals)"""

        # Dataframe to store results
        results = pd.DataFrame(columns=['score', 'params', 'iteration'],
                               index=list(range(self.max_evals)))

        keys, values = zip(*self.param_grid.items())

        i = 0

        # Iterate through every possible combination of hyperparameters
        for v in itertools.product(*values):

            # Create a hyperparameter dictionary
            hyperparameters = dict(zip(keys, v))

            # Set the subsample ratio accounting for boosting type
            hyperparameters['bagging_fraction'] = 1.0 if hyperparameters['boosting_type'] == 'goss' else hyperparameters[
                'bagging_fraction']

            # Evalute the hyperparameters
            eval_results = self.objective(hyperparameters, i)
            print('score:%.5f:' % eval_results[0])

            results.loc[i, :] = eval_results

            i += 1

            # Normally would not limit iterations
            if i > self.max_evals:
                break

        # Sort with best score on top
        results.sort_values('score', ascending=True, inplace=True)
        results.reset_index(inplace=True)

        param_dict = results.loc[0, 'params']
        save_as_json(param_dict, config.LIGHTGBM_BEST_PARAM)
        print(param_dict)
        return results


if __name__ == '__main__':
    config = Config()
    df_train = pd.read_hdf(config.TRAIN_FEATURE_PATH)
    df_train_label = pd.read_hdf(config.LABEL_PATH)
    param_search = ParamSearch(df_train, df_train_label, config)
    param_search.random_search()

