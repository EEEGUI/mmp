import lightgbm as lgb
from sklearn.model_selection import train_test_split


class LGBM:
    def __init__(self, config, train_features, train_labels, test_features):
        self.train_features = train_features
        self.train_labels = train_labels
        self.test_features = test_features
        self.config = config

    def train(self, **kwargs):
        EVAL_SIZE = 0.3
        train_x, val_x, train_y, val_y = train_test_split(self.train_features,
                                                          self.train_labels,
                                                          test_size=EVAL_SIZE,
                                                          shuffle=False)
        lgb_trian = lgb.Dataset(train_x, train_y)
        lgb_eval = lgb.Dataset(val_x, val_y)
        gbm = lgb.train(self.config.PARAM, lgb_trian,
                        num_boost_round=self.config.NUM_BOOST_ROUND,
                        valid_sets=lgb_eval,
                        early_stopping_rounds=self.config.EARLY_STOP_ROUND)

        print('Saving model...')
        # save model to file
        gbm.save_model(self.config.MODEL_SAVING_PATH)

        print('Starting predicting...')
        # predict
        y_pred = gbm.predict(self.test_features, num_iteration=gbm.best_iteration)
        print('finish!')
        return y_pred
    #
