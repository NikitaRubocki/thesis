import pandas as pd
import regex as re
from ensemble.models import ModelFactory


class Ensemble:

    def __init__(self, df, task_type, fast=False):
        self.df = df
        self.task_type = task_type
        self.fast = fast

    def run(self):
        """ Run the ensemble of models and return their collective feature rankings in a dataframe """
        feats = self.df.columns[1:]
        score_df = pd.DataFrame()
        # get a ranking for each model in the ensemble
        for model_class in self.model_registry():
            model = ModelFactory.create_model(model_class, self.df)
            ranking = model.ranking()
            # put rankings into dataframe
            for feat in feats:
                score_df.loc[model_class, feat] = ranking[feat]
        return score_df

    def model_registry(self):
        # pass the task_type into eval to be converted into a function and called
        models = eval(f'self.{self.task_type}()')
        models = self._remove_logistic(models)
        if self.fast:
            # remove slow Sequential models to run the "fast" ensemble
            return list(filter(None, [re.sub(r'.*Sequential.*', '', key) for key in models]))
        return models

    @staticmethod
    def _remove_logistic(models):
        # NASA doesn't like this one, so remove for the ensemble
        return list(filter(None, [re.sub(r'.*Logistic.*', '', key) for key in models]))

    @staticmethod
    def classification():
        """ Return all models with a "C" at the end, standing for classification """
        return [model[0] for model in [re.findall(r'.*C\b', key) for key in ModelFactory.registry.keys()] if model]

    @staticmethod
    def regression():
        """ Return all models with an "R" at the end, standing for regression """
        return [model[0] for model in [re.findall(r'.*R\b', key) for key in ModelFactory.registry.keys()] if model]
