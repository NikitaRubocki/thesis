import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
import seaborn as sns

plot_path = "./plots/"

# Load dataframe
df = pd.read_csv("overall_results.csv")

# Heatmap
# observs = [10, 100, 1000, 10000]
# for obs in observs:
#     obs_df = df.loc[df['obs'] == obs]
#     heatmap_df = obs_df.pivot('feats', 'imp_ratio', 'overlap')
#     # print(heatmap_df)
#     plt.figure()
#     sns.heatmap(heatmap_df, annot=True, cmap="BuPu")
#     plt.savefig(plot_path+f"obs{obs}_heatmap.png")
# print("Heatmaps generated!")

# # 3D Plot
# # axes instance
# fig = plt.figure(figsize=(6,6))
# ax = Axes3D(fig, auto_add_to_figure=False)
# fig.add_axes(ax)

# # get colormap from seaborn
# cmap = ListedColormap(sns.color_palette("husl", 256).as_hex())

# # plot
# sc = ax.scatter(df['feats'], df['imp_ratio'], df['overlap'], c=df['overlap'], cmap=cmap, alpha=1)
# ax.set_xlabel('# of Features')
# ax.set_ylabel('Importance Ratio')
# ax.set_zlabel('Overlap')

# # legend
# plt.legend(*sc.legend_elements(), bbox_to_anchor=(1.05, 1), loc=2)

# # save
# plt.savefig(plot_path+"scatter_plot.png", bbox_inches='tight')
# print("Scatterplot generated!")
