import numpy as np 
import pandas as pd
from sklearn.datasets import make_regression


class Dataset:

    def generate(feat, imp, obs, rand=21):
        # create random data
        X, y, coef = make_regression(
            n_features=feat,
            n_informative=imp,
            n_samples=obs,
            noise=0.0,
            random_state=rand,
            coef=True
        )

        # put data in dataframe
        df = pd.concat([pd.Series(y), pd.DataFrame(X)], axis=1)
        df.columns = ["label"] + [f"x{i}" for i in range(feat)]

        # adjust col names based on important features (coef)
        adj_col_names = {}
        col_names = list(df.columns)
        col_names.pop(0)
        for idx in range(feat):
            if coef[idx] == 0:
                continue
            adj_col_names[col_names[idx]] = f"imp{idx}"
        df.rename(columns=adj_col_names, inplace=True)

        # save dataset
        fname = f"feat{feat}_imp{imp}_obs{obs}.csv"
        df.to_csv(f"datasets/{fname}", index=False)
