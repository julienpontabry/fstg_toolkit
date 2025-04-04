from math import ceil

from dash import html, Input, Output, callback, dcc
from plotly import graph_objects as go
from plotly.subplots import make_subplots


def build_matrices_figure(corr, n_cols=5):
    # create figure for matrices to display
    names = list(corr.keys())
    n_rows = ceil(len(corr) / n_cols)
    fig = make_subplots(rows=n_rows, cols=n_cols)

    for i, name in enumerate(names):
        row = i // n_cols + 1
        col = i % n_cols + 1
        hm = go.Heatmap(
            z=(corr[name][0]),
            zmin=-1, zmax=1,
            colorscale="RdBu_r",
            showscale=False
        )
        fig.add_trace(hm, row=row, col=col)

    # set up the layout
    fig.update_layout(width=200 * n_cols, height=200 * n_rows)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False, autorange='reversed')

    for i in range(1, n_rows + 1):
        for j in range(1, n_cols + 1):
            fig.update_yaxes(scaleanchor=f"x{(i-1) * n_cols + j}", scaleratio=1, row=i, col=j)
            fig.update_xaxes(constrain="domain", row=i, col=j)

    return fig


layout = [
    html.Div([
        dcc.Graph(figure={}, id='mtx-graph')
    ])
]

@callback(
    Output('mtx-graph', 'figure'),
    Input('store-corr', 'data'),
)
def update_figure(corr):
    if corr is None:
        return {}

    return build_matrices_figure(corr)
