import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
#from silkscreen.prediction.preprocessor import Preprocessor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


class TrainPreprocessor():
    """ """

    def __init__(self, config) -> None:
        self.target_encode = config.get("target_encode")
        self.numeric_scale = config.get("numeric", {}).get("scale")
        self.numeric_raw = config.get("numeric", {}).get("raw")
        self.categorical = config.get("categorical")
        self.ordinal = config.get("ordinal")
        self.target = config.get("target")
        self.column_transformer = None

    def define_transformers(self, data: pd.DataFrame):

        transformers = []

        if self.numeric_scale:
            numeric_scale_pipeline = Pipeline(steps=[
                ('impute', SimpleImputer(strategy='median')),
                ('scale', MinMaxScaler())
            ])
            transformers.append(('numeric_scale', numeric_scale_pipeline, self.numeric_scale))
        if self.numeric_raw:
            numeric_raw_pipeline = Pipeline(steps=[
                ('impute', SimpleImputer(strategy='median'))
            ])
            transformers.append(('numeric_raw', numeric_raw_pipeline, self.numeric_raw))
        if self.target_encode:
            target_encoder = (
                data.groupby(self.target_encode)[self.target].mean().to_dict()
            )
            #data[self.target_encode] = data[self.target_encode].map(self.target_encoder)
        if self.categorical:
            categorical_pipeline = Pipeline(steps=[
                ('impute', SimpleImputer(strategy='most_frequent')),
                ('one-hot', OneHotEncoder(min_frequency=0.1, handle_unknown='ignore', sparse=False))
            ])
            transformers.append(('categorical', categorical_pipeline, self.categorical))

        print("Transformers", transformers)

        self.column_transformer = ColumnTransformer(transformers)

    def fit_preprocessor(self, data: pd.DataFrame):

        return self.column_transformer.fit_transform(data)

    def transform_columns(self, data: pd.DataFrame):

        return self.column_transformer.transform(data)


