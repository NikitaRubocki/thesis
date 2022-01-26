import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from treeinterpreter import treeinterpreter as ti
from sklearn.model_selection import train_test_split
import pandas as pd
from abc import abstractmethod
from .model import Model


class Error(Model):

    @abstractmethod
    def __init__(self, df):
        """ Each subclass should implement its own model as the default """
        super().__init__(df)
        self.model = None

    def _metric(self):
        X, y = self.df.iloc[:, 1:], self.df.iloc[:, 0]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
        self.model.fit(X_train, y_train)
        sample_df = X_test.loc[self._error(X_test, y_test)].values.reshape(1, -1)
        contributions = self._contributions(sample_df)
        return abs(contributions[0].round(12))

    def _contributions(self, sample_df):
        """ Obtain the contributions from TreeInterpreter """
        _, _, contributions = ti.predict(self.model, sample_df)
        return contributions

    @abstractmethod
    def _error(self, X, y):
        """ Each subclass should implement its own pred_diff sorting to get the proper index as the return value """
        pred_diff = pd.DataFrame()
        pred_diff['difference'] = abs(y - self.model.predict(X))
        return pred_diff


class SmallError(Error):

    @abstractmethod
    def __init__(self, df):
        """ Each subclass should implement its own model as the default """
        super().__init__(df)
        self.model = None

    def _error(self, X, y):
        pred_diff = super()._error(X, y)
        return pred_diff.sort_values('difference').head(1).index.values[0]


class BigError(Error):

    @abstractmethod
    def __init__(self, df):
        """ Each subclass should implement its own model as the default """
        super().__init__(df)
        self.model = None

    def _error(self, X, y):
        pred_diff = super()._error(X, y)
        return pred_diff.sort_values('difference', ascending=False).head(1).index.values[0]


class SmallErrorRandomForestR(SmallError):

    def __init__(self, df):
        super().__init__(df)
        self.model = RandomForestRegressor(random_state=1, n_jobs=-1)


class BigErrorRandomForestR(BigError):

    def __init__(self, df):
        super().__init__(df)
        self.model = RandomForestRegressor(random_state=1, n_jobs=-1)


class SmallErrorRandomForestC(SmallError):

    def __init__(self, df):
        super().__init__(df)
        self.model = RandomForestClassifier(random_state=1, n_jobs=-1)

    def _contributions(self, sample_df):
        """ Classification forests need to average the contributions per target label """
        _, _, contributions = ti.predict(self.model, sample_df)
        contributions = np.asarray([[contrib.mean() for contrib in abs(contributions[0])]])
        return contributions


class BigErrorRandomForestC(BigError):

    def __init__(self, df):
        super().__init__(df)
        self.model = RandomForestClassifier(random_state=1, n_jobs=-1)

    def _contributions(self, sample_df):
        """ Classification forests need to average the contributions per target label """
        _, _, contributions = ti.predict(self.model, sample_df)
        contributions = np.asarray([[contrib.mean() for contrib in abs(contributions[0])]])
        return contributions
