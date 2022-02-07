from generate_dataset import Dataset
import math
import os
import re

feats = [3, 5, 10, 15, 25, 50, 100]
imp_ratios = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
observs = [10, 100, 1000, 10000]

# Create datasets
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

# Gather datasets in dir
datasets = []
for root, dirs, files in os.walk("./datasets"):
	for file in files:
		datasets.append(os.path.join(root, file))

# Gather results in dir
results = []
for root, dirs, files in os.walk("./results"):
	for file in files:
		results.append(os.path.join(root, file))

# Run experiments
print(f"Running {len(datasets)} experiments...")
for data in datasets:
    print("DATA:", data)
    output = f"./results/{data[11:-4]}.json"
    if output in results:
        print("Skipping")
        continue
    os.system(f"python3 ./feature_analysis/main.py {data} -t regression -vf -j {output}")
    print("\n")
print("Experiments complete!")
