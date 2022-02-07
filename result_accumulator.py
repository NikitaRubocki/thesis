import os
import re
import json
import pandas as pd
import pprint

# Gather results in dir
results = []
for root, dirs, files in os.walk("./results"):
	for file in files:
		results.append(os.path.join(root, file))

# Setup dataframe
df_cols = ['feats', 'imp_ratio', 'obs', 'sum_overlap', 'sum_distance', 'avg_overlap', 'avg_distance']
df = pd.DataFrame(columns=df_cols)

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

# CHANGE THIS LATER
results = results[17:]

count = 0
for res in results:
    print("RESULT:", res)
    # get feat, imp, and obs out of file name
    feat, imp, obs = get_values(res)

    # bring in json data
    data = json.load(open(res, 'r'))
    # pprint.pprint(data)

    # calculate overlap
    sum_olap = get_overlap(data['sum'], imp)
    avg_olap = get_overlap(data['average'], imp)
    print(sum_olap, avg_olap)

    # calculate distance

    # df.loc[len(df.index)] = [feat, imp, obs, sum_olap, sum_dist, avg_olap, avg_dist]
    count += 1
    if count == 3:
        break

