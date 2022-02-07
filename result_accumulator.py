import os
import re
import json
import pandas as pd
import pprint
import statistics as st
import matplotlib.pyplot as plt
import seaborn as sns

def get_values(res):
    vals = res[10:-5].split("_")
    feat = int(vals[0][4:])
    imp = int(vals[1][3:])
    obs = int(vals[2][3:])
    return feat, imp, obs

def get_overlap(data, imp):
    keys = list(data.keys())
    imp_count = 0
    # count how many imps appear in the first "imp" elements of the list
    for i in range(imp):
        if 'imp' in keys[i]:
            imp_count += 1
    return round(imp_count/imp, 2)

def get_distance(data):
    # break data into imps and other
    imps, other = {}, {}
    for key, val in data.items():
        if 'imp' in key:
            imps[key] = val
        else:
            other[key] = val

    # calculate mean and subtract
    imp_mean = st.mean(imps.values())
    other_mean = st.mean(other.values())
    dist = abs(imp_mean - other_mean)

    # calculate range and standardize
    keys = list(data.keys())
    data_range = data[keys[0]] - data[keys[-1]]
    return round(dist/data_range, 4)


# Gather results in dir
results = []
for root, dirs, files in os.walk("./results"):
	for file in files:
		results.append(os.path.join(root, file))

# Setup dataframe
df_cols = ['res_file', 'feats', 'imp_ratio', 'obs', 'overlap', 'distance']
df = pd.DataFrame(columns=df_cols)

# Do some calculations
# calculations are the same for sum and avg, so take sum
count = 0
gold_imps = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
for res in results:
    data = json.load(open(res, 'r'))

    # get feat, imp, and obs out of file name
    feat, imp, obs = get_values(res)
    imp_ratio = round(imp/feat, 2)
    if imp_ratio not in gold_imps:
        if imp_ratio < 0.03:
            imp_ratio = 0.01
        elif 0.03 <= imp_ratio < 0.075:
            imp_ratio = 0.05
        elif 0.075 <= imp_ratio < 0.175:
            imp_ratio = 0.1
        elif 0.175 <= imp_ratio < 0.375:
            imp_ratio = 0.25
        elif 0.3735 <= imp_ratio < 0.625:
            imp_ratio = 0.5
        elif 0.625 <= imp_ratio < 0.825:
            imp_ratio = 0.75
        elif 0.825 <= imp_ratio < 0.95:
            imp_ratio = 0.9
        elif 0.95 <= imp_ratio:
            imp_ratio = 0.99
        else:
            imp_ratio = None

    if feat == imp:
        continue

    # calculate overlap and distance
    olap = get_overlap(data['sum'], imp)
    dist = get_distance(data['sum'])

    # save to df
    df.loc[len(df.index)] = [res[10:], feat, imp_ratio, obs, olap, dist]

# df.to_csv('overall_results.csv', index=False)
# print("Dataframe saved!")

# Plot time!!
# print(df['imp_ratio'])
# print(df['imp_ratio'].value_counts())
# print(df)


# print(df.loc[df['obs'] == 100])
obs_df = df.loc[df['obs'] == 100]
print(df['feats'].value_counts())
obs_df = obs_df.loc[obs_df['feats'] == 5]
print(obs_df)
# print(obs_df['imp_ratio'].unique())

mapping = obs_df.pivot('feats', 'imp_ratio', 'distance')
# print(mapping)

# sns.heatmap(mapping, cmap="BuPu")
# plt.savefig("heatmap.png")
