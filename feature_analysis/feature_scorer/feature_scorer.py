from ensemble import Ensemble


class FeatureScorer:

    def __init__(self, df, task_type, fast=False):
        self.df = df
        self.task_type = task_type
        self.fast = fast

    def importance_scores(self):
        """ Given a dataframe, returns a dictionary with sum and average where (key, val) = (feature, score) """
        self.df = Ensemble(self.df, self.task_type, self.fast).run()
        return self._calculate_scores()

    def _calculate_scores(self):
        return {
            'sum': self._summation(),
            'average': self._average()
        }

    def _summation(self):
        sums = {}
        for feat in self.df.columns:
            sums[feat] = self.df[feat].sum()
        return sums

    def _average(self):
        avgs = {}
        for feat in self.df.columns:
            avgs[feat] = round(self.df[feat].mean(), 4)
        return avgs
