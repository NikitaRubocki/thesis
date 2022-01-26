from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from xgboost import XGBRegressor, XGBClassifier
import numpy as np
from abc import abstractmethod
from .model import Model
from .sklearn_sequential import SequentialFS


class Sequential(Model):
    """ This class works slightly differently in its metric. Instead of returning continuous values,
        a boolean mask is returned where either a feature was selected (i.e. important) or it was not.
        How many features are selected is determined by `n_features_to_select` """

    @abstractmethod
    def __init__(self, df):
        """ Each subclass should implement its own model as the default """
        super().__init__(df)
        self.model = None

    def _metric(self):
        X, y = self.df.iloc[:, 1:], self.df.iloc[:, 0]
        sfs = SequentialFS(estimator=self.model, n_features_to_select=0.5,
                           cv=3, direction=self._direction(), n_jobs=-1)
        sfs.fit(X, y)
        return np.asarray(sfs.get_support(), dtype=int)

    @abstractmethod
    def _direction(self):
        """ Each subclass should implement its own direction as the return value """
        pass


class ForwardSequential(Sequential):

    @abstractmethod
    def __init__(self, df):
        """ Each subclass should implement its own model as the default """
        super().__init__(df)
        self.model = None

    def _direction(self):
        return 'forward'


class BackwardSequential(Sequential):

    @abstractmethod
    def __init__(self, df):
        """ Each subclass should implement its own model as the default """
        super().__init__(df)
        self.model = None

    def _direction(self):
        return 'backward'


class ForwardSequentialRandomForestR(ForwardSequential):

    def __init__(self, df):
        super().__init__(df)
        self.model = RandomForestRegressor(random_state=1, n_jobs=-1)


class BackwardSequentialRandomForestR(BackwardSequential):

    def __init__(self, df):
        super().__init__(df)
        self.model = RandomForestRegressor(random_state=1, n_jobs=-1)


class ForwardSequentialXGBoostR(ForwardSequential):

    def __init__(self, df):
        super().__init__(df)
        self.model = XGBRegressor(random_state=1, n_jobs=-1)


class BackwardSequentialXGBoostR(BackwardSequential):

    def __init__(self, df):
        super().__init__(df)
        self.model = XGBRegressor(random_state=1, n_jobs=-1)


class ForwardSequentialLinearRegressorR(ForwardSequential):

    def __init__(self, df):
        super().__init__(df)
        self.model = LinearRegression()


class BackwardSequentialLinearRegressorR(BackwardSequential):

    def __init__(self, df):
        super().__init__(df)
        self.model = LinearRegression()


class ForwardSequentialRandomForestC(ForwardSequential):

    def __init__(self, df):
        super().__init__(df)
        self.model = RandomForestClassifier(random_state=1, n_jobs=-1)


class BackwardSequentialRandomForestC(BackwardSequential):

    def __init__(self, df):
        super().__init__(df)
        self.model = RandomForestClassifier(random_state=1, n_jobs=-1)


class ForwardSequentialXGBoostC(ForwardSequential):

    def __init__(self, df):
        super().__init__(df)
        self.model = XGBClassifier(random_state=1, n_jobs=-1, use_label_encoder=False, verbosity=0)


class BackwardSequentialXGBoostC(BackwardSequential):

    def __init__(self, df):
        super().__init__(df)
        self.model = XGBClassifier(random_state=1, n_jobs=-1, use_label_encoder=False, verbosity=0)


class ForwardSequentialLogisticRegressorC(ForwardSequential):

    def __init__(self, df):
        super().__init__(df)
        self.model = LogisticRegression(max_iter=10000)


class BackwardSequentialLogisticRegressorC(BackwardSequential):

    def __init__(self, df):
        super().__init__(df)
        self.model = LogisticRegression(max_iter=10000)
