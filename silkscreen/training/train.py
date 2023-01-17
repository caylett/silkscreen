import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, log_loss, precision_score, recall_score, roc_auc_score
from silkscreen.training.train_preprocessor import TrainPreprocessor
import xgboost as xgb
#import lightgbm as lgbm
import pickle
from datetime import date
import uuid
import os

class Train:
    def __init__(self, feature_config, train_config) -> None:
        self.test_proportion = train_config['test_proportion']
        self.model_type = train_config['model_type']
        self.preprocessor = TrainPreprocessor(feature_config)
        self.target = feature_config['target']
        self.use_case_name = train_config['use_case_name']
    
    def train_model(self, data: pd.DataFrame):
        """
        """

        holdout = data.sample(frac=0.1, random_state = 111)
        # y = data[self.target].copy()

        train_data, test_data = train_test_split(data, test_size=self.test_proportion)

        y_train = train_data[self.target].copy()
        y_test = test_data[self.target].copy()
        y_eval = holdout[self.target].copy()

        self.preprocessor.define_transformers(train_data)
        X_train = self.preprocessor.fit_preprocessor(train_data)
        
        # X_train = self.preprocessor.tranform(X_train)
        X_test = self.preprocessor.transform_columns(test_data)
        X_eval = self.preprocessor.transform_columns(holdout)
        
        model = self._fit_model(X_train, X_test, y_train, y_test)

        score = self._score_model(model, X_eval, y_eval)

        path_name = "./models/{}/{}/".format(self.use_case_name, date.today())
        file_name = "{}.pkl".format(uuid.uuid4())

        if not os.path.exists(path_name):
            os.makedirs(path_name)

        pickle.dump(model, open(path_name + file_name, 'wb'))
        
        return score, model

    def _fit_model(self, X_train, X_test, y_train, y_test):
        
        if self.model_type=='xgboostclassifier':
            model = xgb.XGBClassifier(n_estimators=100, max_depth=6, early_stopping_rounds=20)
            model.fit(X_train, y_train, 
                eval_set=[(X_train, y_train), (X_test, y_test)])

        if self.model_type=='xgboostregressor':
            model = xgb.XGBRegressor(n_estimators=100, max_depth=6, early_stopping_rounds=20)
            model.fit(X_train, y_train, 
                eval_set=[(X_train, y_train), (X_test, y_test)])    

        return model

    def _score_model(self, model, X_eval, y_eval):

        score = {}
        y_pred = model.predict(X_eval)

        if self.model_type=='xgboostclassifier':
            score['log_loss'] = log_loss(y_eval, y_pred)
            score['precision'] = precision_score(y_eval, y_pred)
            score['recall'] = recall_score(y_eval, y_pred)
            score['auc'] = roc_auc_score(y_eval, y_pred)
        if self.model_type=='xgboostregressor':
            mse = mean_squared_error(y_eval, y_pred)
            score['mse'] = mse
            score['rmse'] = np.sqrt(mse)
            score['mae'] = mean_absolute_error(y_eval, y_pred)
            score['r2'] = r2_score(y_eval, y_pred)

        return score


