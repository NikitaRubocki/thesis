from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from xgboost import XGBRegressor, XGBClassifier
from sklearn.base import clone
from abc import abstractmethod
from .model import Model


class DropCol(Model):

    @abstractmethod
    def __init__(self, df):
        """ Each subclass should implement its own model as the default """
        super().__init__(df)
        self.model = None

    def _metric(self):
        def drop_col(model, X_train, y_train):
            model_clone = clone(model)
            model_clone.fit(X_train, y_train)
            benchmark_score = round(model_clone.score(X_train, y_train), 12)
            diffs = []
            # iterate over all columns and store feature importance (difference between benchmark and new model)
            for col in X_train.columns:
                model_clone = clone(model)
                model_clone.fit(X_train.drop(col, axis=1), y_train)
                drop_col_score = round(model_clone.score(X_train.drop(col, axis=1), y_train), 12)
                diffs.append(benchmark_score - drop_col_score)

            return diffs

        X, y = self.df.iloc[:, 1:], self.df.iloc[:, 0]
        return drop_col(self.model, X, y)


class DropColRandomForestR(DropCol):

    def __init__(self, df):
        super().__init__(df)
        self.model = RandomForestRegressor(random_state=1, n_jobs=-1)


class DropColXGBoostR(DropCol):

    def __init__(self, df):
        super().__init__(df)
        self.model = XGBRegressor(random_state=1, n_jobs=-1)


class DropColLinearRegressorR(DropCol):

    def __init__(self, df):
        super().__init__(df)
        self.model = LinearRegression()


class DropColRandomForestC(DropCol):

    def __init__(self, df):
        super().__init__(df)
        self.model = RandomForestClassifier(random_state=1, n_jobs=-1)


class DropColXGBoostC(DropCol):

    def __init__(self, df):
        super().__init__(df)
        self.model = XGBClassifier(random_state=1, n_jobs=-1, use_label_encoder=False, verbosity=0)


class DropColLogisticRegressorC(DropCol):

    def __init__(self, df):
        super().__init__(df)
        self.model = LogisticRegression(max_iter=10000)
