from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from xgboost import XGBRegressor, XGBClassifier
from rfpimp import permutation_importances
from sklearn.metrics import r2_score
from abc import abstractmethod
from .model import Model


class Permutation(Model):

    @abstractmethod
    def __init__(self, df):
        """ Each subclass should implement its own model as the default """
        super().__init__(df)
        self.model = None

    def _metric(self):
        def r2(model, X_score, y_score):
            return round(r2_score(y_score, model.predict(X_score)), 12)

        X, y = self.df.iloc[:, 1:], self.df.iloc[:, 0]
        return permutation_importances(self.model, X, y, r2)['Importance']


class PermutationRandomForestR(Permutation):

    def __init__(self, df):
        super().__init__(df)
        self.model = RandomForestRegressor(random_state=1, n_jobs=-1)


class PermutationXGBoostR(Permutation):

    def __init__(self, df):
        super().__init__(df)
        self.model = XGBRegressor(random_state=1, n_jobs=-1)


class PermutationLinearRegressorR(Permutation):

    def __init__(self, df):
        super().__init__(df)
        self.model = LinearRegression()


class PermutationRandomForestC(Permutation):

    def __init__(self, df):
        super().__init__(df)
        self.model = RandomForestClassifier(random_state=1, n_jobs=-1)


class PermutationXGBoostC(Permutation):

    def __init__(self, df):
        super().__init__(df)
        self.model = XGBClassifier(random_state=1, n_jobs=-1, use_label_encoder=False, verbosity=0)


class PermutationLogisticRegressorC(Permutation):

    def __init__(self, df):
        super().__init__(df)
        self.model = LogisticRegression(max_iter=10000)
