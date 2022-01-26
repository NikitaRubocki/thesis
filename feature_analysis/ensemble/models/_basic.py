from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from xgboost import XGBRegressor, XGBClassifier
from abc import abstractmethod
from .model import Model


class Basic(Model):

    @abstractmethod
    def __init__(self, df):
        """ Each subclass should implement its own model as the default """
        super().__init__(df)
        self.model = None

    def _metric(self):
        X, y = self.df.iloc[:, 1:].values, self.df.iloc[:, 0].values
        self.model.fit(X, y)
        return self._weights()

    @abstractmethod
    def _weights(self):
        """ Each subclass should implement its own weight attribute as the default """
        pass


class RandomForestR(Basic):

    def __init__(self, df):
        super().__init__(df)
        self.model = RandomForestRegressor(random_state=1, n_jobs=-1)

    def _weights(self):
        return self.model.feature_importances_


class XGBoostR(Basic):

    def __init__(self, df):
        super().__init__(df)
        self.model = XGBRegressor(random_state=1, n_jobs=-1)

    def _weights(self):
        return self.model.feature_importances_


class LinearRegressorR(Basic):

    def __init__(self, df):
        super().__init__(df)
        self.model = LinearRegression()

    def _weights(self):
        return abs(self.model.coef_)


class RandomForestC(Basic):

    def __init__(self, df):
        super().__init__(df)
        self.model = RandomForestClassifier(random_state=1, n_jobs=-1)

    def _weights(self):
        return self.model.feature_importances_


class XGBoostC(Basic):

    def __init__(self, df):
        super().__init__(df)
        self.model = XGBClassifier(random_state=1, n_jobs=-1, use_label_encoder=False, verbosity=0)

    def _weights(self):
        return self.model.feature_importances_


class LogisticRegressorC(Basic):

    def __init__(self, df):
        super().__init__(df)
        self.model = LogisticRegression(max_iter=10000)

    def _weights(self):
        return abs(self.model.coef_[0])
