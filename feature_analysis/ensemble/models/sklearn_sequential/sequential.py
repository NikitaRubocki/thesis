from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import cross_val_score
import numpy as np


class SequentialFS(SequentialFeatureSelector):

    def _get_best_new_feature(self, estimator, X, y, current_mask):
        # Return the best new feature to add to the current_mask, i.e. return
        # the best new feature to add (resp. remove) when doing forward
        # selection (resp. backward selection)
        candidate_feature_indices = np.flatnonzero(~current_mask)
        scores = {}
        for feature_idx in candidate_feature_indices:
            candidate_mask = current_mask.copy()
            candidate_mask[feature_idx] = True
            if self.direction == 'backward':
                candidate_mask = ~candidate_mask
            X_new = X[:, candidate_mask]
            scores[feature_idx] = cross_val_score(
                estimator, X_new, y, cv=self.cv, scoring=self.scoring,
                n_jobs=self.n_jobs).mean()
        # THIS NEXT LINE IS THE ONLY EDIT
        # needed because without rounding the ensemble doesn't produce consistent results because floats are a**holes
        # otherwise this is a direct copy of sklearn's SequentialFeatureSelector
        scores = [round(x, 12) for x in scores]
        return max(scores, key=lambda idx: scores[idx])
