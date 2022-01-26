from abc import ABC, abstractmethod
import inspect
import numpy as np
np.random.seed(21)


class ModelFactory:
    """ Automatically registers subclasses of Model for use in the Ensemble class """
    registry = {}

    @classmethod
    def register(cls, model):
        if model not in cls.registry and inspect.isabstract(model) is False:
            cls.registry[model.__name__] = model

    @classmethod
    def create_model(cls, name, *args):
        model = cls.registry[name]
        return model(*args)


class Model(ABC):

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        ModelFactory.register(cls)

    @abstractmethod
    def __init__(self, df):
        """ Each subclass should implement its own model as the default """
        """ The target label is assumed to be the first column in the dataframe """
        self.model = None
        self.df = df

    def ranking(self):
        """ Return the ranking of each feature from a model based on its internal importances """
        return self._scores(self._importances(), self._max_score())

    @staticmethod
    def _scores(imps, max_score):
        """ Give each feature a score based on its importance ranking """
        scores = {}
        for key in imps:
            # if importance is zero, don't give a score
            if imps[key] == 0.0:
                scores[key] = 0
                continue
            scores[key] = max_score
            max_score -= 1
        return scores

    def _importances(self):
        """ Gather importances in a dictionary and order them high -> low """
        values = self._metric()
        indices = np.argsort(values)[::-1]
        feats = self._feats()
        importances = {}
        for i in range(len(feats)):
            importances[feats[indices[i]]] = values[indices[i]]
        return importances

    @abstractmethod
    def _metric(self):
        """ Each subclass should implement its own metric as the default """
        pass

    def _max_score(self):
        """ Return the max score, which is the amount of features available - 1 (allowing for scores of 0.0) """
        return len(self._feats()) - 1

    def _feats(self):
        """ Return the features of the dataset (label should be the first column) """
        return self.df.columns[1:]
