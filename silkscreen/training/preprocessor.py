import pandas as pd
from sklearn.preprocessing import OneHotEncoder


class TrainPreprocessor:
    """
    """
    def __init__(self, config) -> None:
        self.target_encode = config.get("target_encode")
        self.numeric = config.get("numeric")

    def train_preprocessor(self, data):
        
        data = self._drop_na(data)

        if self.numeric:
            pass
        if self.target_encode:
            pass

        return data

    def _drop_na(self, data):
        return data.drop_na()

    def _scale_numeric(self):
        pass

    def _target_encode(self):
        pass

    def _ordinal_encode(self):
        pass

    def _one_hot_encode(self, X):
        OneHotEncoder(min_frequency=0.1, handle_unknown="ignore").fit(X)

