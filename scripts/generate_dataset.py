import numpy as np 
import pandas as pd
import math
from sklearn.datasets import make_regression


class Dataset:

    # experiment02_07_22 seed: 21
    # experiment02_21_22 seed: 42
    def generate(feat, imp, imp_ratio, obs, rand=42):
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
        fname = f"feat{feat}_imp{imp}_rat{imp_ratio}_obs{obs}.csv"
        df.to_csv(f"datasets/{fname}", index=False)

if __name__ == '__main__':
    # # experiment 02_07_22
    # feats = [3, 5, 10, 15, 25, 50, 75, 100]
    # imp_ratios = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    # observs = [10, 100, 1000, 10000]

    # experiment 02_21_22
    # feats = [3, 5, 10, 15, 25, 50, 75, 100, 150]
    # imp_ratios = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    # observs = [10, 100, 1000, 2000, 5000, 7500, 10000]

    # experiment 02_25_22 for p value test
    feats = list(range(10, 140, 20))
    imp_ratios = [x/100 for x in list(range(5, 100, 15))]
    observs = list(range(100, 10001, 1500))
    print(feats)
    print(imp_ratios)
    print(observs)

    # # Create datasets
    # print("Generating datasets...")
    # count = 0
    # for feat in feats:
    #     for rat in imp_ratios:
    #         adj_imp = math.ceil(rat*feat)
    #         if adj_imp == feat:
    #             continue
    #         for obs in observs:
    #             if feat >= obs:
    #                 continue
    #             Dataset.generate(feat=feat, imp=adj_imp, imp_ratio=rat, obs=obs)
    #             count += 1
    # print(f"Dataset generation complete. {count} datasets created.\n")