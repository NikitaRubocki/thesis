import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
import os


# Get and save dataset
df = fetch_california_housing(as_frame=True)
target = df.frame['MedHouseVal']
df.frame.drop(labels=['MedHouseVal'], axis=1,inplace = True)
df.frame.insert(0, 'target', target)
# print(df.frame.head())
df.frame.to_csv('california_housing.csv', index=False)

# Run dataset
data = 'california_housing.csv'
output = 'california_housing.json'
os.system(f"python3 ../feature_analysis/main.py {data} -t regression -vf -j {output}")