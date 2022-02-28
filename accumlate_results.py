import os
import re
import sys
import json
import pandas as pd
import statistics as st

def get_values(res):
    vals = res[10:-5].split("_")
    feat = int(vals[0][4:])
    imp = int(vals[1][3:])
    imp_ratio = float(vals[2][3:])
    obs = int(vals[3][3:])
    return feat, imp, imp_ratio, obs

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
for res in results:
    data = json.load(open(res, 'r'))

    # get feat, imp, and obs out of file name
    feat, imp, imp_ratio, obs = get_values(res)

    # calculate overlap and distance
    olap = get_overlap(data['sum'], imp)
    dist = get_distance(data['sum'])

    # save to df
    df.loc[len(df.index)] = [res[10:], feat, imp_ratio, obs, olap, dist]

df.to_csv('overall_results.csv', index=False)
print("Dataframe saved!")
