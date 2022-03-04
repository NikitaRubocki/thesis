import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt

# Load dataframe
df = pd.read_csv("./experiment02_21_22/overall_results.csv")

g = sns.FacetGrid(df, col='feats', hue='obs')
g.map(sns.lineplot, 'imp_ratio', 'overlap')
g.add_legend()
g.savefig('facet_grid_imp_olap21.png')
