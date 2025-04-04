from math import ceil

from dash import Input, Output, callback, dcc, html
from dash.exceptions import PreventUpdate
from plotly import graph_objects as go
from plotly.subplots import make_subplots


def build_matrices_figure(corr, t, n_cols=5):
    # create figure for matrices to display
    names = list(corr.keys())
    n_rows = ceil(len(corr) / n_cols)
    fig = make_subplots(rows=n_rows, cols=n_cols)

    for i, name in enumerate(names):
        row = i // n_cols + 1
        col = i % n_cols + 1
        hm = go.Heatmap(
            z=(corr[name][t]),
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
        dcc.Slider(min=0, max=1, step=1, value=0, id='mtx-slider-time'),
        dcc.Graph(figure={}, id='mtx-graph')
    ])
]


@callback(
    Output('mtx-graph', 'figure'),
    Output('mtx-slider-time', 'max'),
    Output('mtx-slider-time', 'marks'),
    Input('store-corr', 'data'),
    Input('mtx-slider-time', 'value'),
)
def update_figure(corr, slider_value):
    if corr is None or len(corr) == 0:
        raise PreventUpdate

    max_slider_value = len(next(iter(corr.values()))) - 1
    marks_slider = {i: str(i) for i in range(0, max_slider_value + 1, max_slider_value//10)}
    return build_matrices_figure(corr, slider_value), max_slider_value, marks_slider
