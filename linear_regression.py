import pandas as pd
import statsmodels.api as sm

# Load dataframe
df = pd.read_csv("overall_results.csv")

# Split the data
# Y = df['overlap']
Y = df['distance']
X = df[['feats', 'imp_ratio', 'obs']]
X = sm.add_constant(X)
# print(X)

# Run LR and show results
lr = sm.OLS(Y, X)
lr_res = lr.fit()
print(lr_res.summary())