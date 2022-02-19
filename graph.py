import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
import seaborn as sns

plot_path = "./plots/"

# Load dataframe
df = pd.read_csv("overall_results.csv")

# Heatmap
observs = [10, 100, 1000, 10000]
for obs in observs:
    obs_df = df.loc[df['obs'] == obs]
    heatmap_df = obs_df.pivot('feats', 'imp_ratio', 'overlap')
    plt.figure()
    plt.title("Overlap")
    sns.heatmap(heatmap_df, annot=True, cmap="BuPu")
    plt.savefig(plot_path+f"obs{obs}_olap_heatmap.png")
print("Overlap heatmaps generated!")

for obs in observs:
    obs_df = df.loc[df['obs'] == obs]
    heatmap_df = obs_df.pivot('feats', 'imp_ratio', 'distance')
    plt.figure()
    plt.title("Distance")
    sns.heatmap(heatmap_df, annot=True, cmap="BuPu")
    plt.savefig(plot_path+f"obs{obs}_dist_heatmap.png")
print("Distance heatmaps generated!")

# 3D Plot
# axes instance
fig = plt.figure(figsize=(6,6))
ax = Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)

# get colormap from seaborn
cmap = ListedColormap(sns.color_palette("husl", 256).as_hex())

# plot
sc = ax.scatter(df['feats'], df['imp_ratio'], df['obs'], c=df['overlap'], cmap=cmap, alpha=1)
ax.set_xlabel('# of Features')
ax.set_ylabel('Importance Ratio')
ax.set_zlabel('Overlap')

# legend
plt.legend(*sc.legend_elements(), bbox_to_anchor=(1.05, 1), loc=2)

# save
plt.savefig(plot_path+"scatter_plot.png", bbox_inches='tight')
print("Scatterplot generated!")
