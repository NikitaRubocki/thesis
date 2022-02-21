import math
import os
import re

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
    print("Running dataset: ", data)
    output = f"./results/{data[11:-4]}.json"
    if output in results:
        print("Skipping")
        continue
    os.system(f"python3 ./feature_analysis/main.py {data} -t regression -vf -j {output}")
    print("\n")
print("Experiments complete!")
