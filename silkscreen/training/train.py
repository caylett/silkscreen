import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from silkscreen.training.preprocessor import TrainPreprocessor
import xgboost as xgb
import lightgbm as lgbm

class Train:
    def __init__(self, config) -> None:
        self.test_proportion = config['test_proportion']
        self.model_type = config['model_type']
        self.preprocessor = TrainPreprocessor()
    
    def train_model(self, data):
        data = self.preprocessor.train_preprocessor(data)
        return data

    def split_data(self, data):
        X_train, X_test, y_train, y_test = train_test_split(data, test_size=self.test_proportion)
        return X_train, X_test, y_train, y_test

    def fit_model(self):
        
        if 'xgboostclassifier' in self.model_type:
            xgb_clf = xgb.XGBClassifier()
            xgb_clf.fit()

        if 'xgboostregressor' in self.model_type:
            xbg_rg = xgb.XGBRegressor()
            xgb_rg.fit()







        return model
