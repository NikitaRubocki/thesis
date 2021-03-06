import pandas as pd
import plotly.express as px

# Load dataframe
df = pd.read_csv("overall_results.csv")

fig = px.scatter_3d(
    df, 
    # x='feats', 
    # y='imp_ratio', 
    # z='obs',
    # color='overlap', 
    # color='distance', 
    x='feats', 
    y='obs', 
    z='overlap',
    color='imp_ratio',
)
fig.update_layout(margin=dict(l=0.2, r=0.2, b=0.2, t=0.2))


import dash
from dash import dcc
from dash import html

app = dash.Dash()
app.layout = html.Div([
    dcc.Graph(figure=fig)
])

app.run_server(debug=True, use_reloader=False)