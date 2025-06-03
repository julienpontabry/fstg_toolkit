from math import ceil

from dash.exceptions import PreventUpdate
from dash_extensions.enrich import Input, Output, State, callback, dcc
import dash_bootstrap_components as dbc
from plotly import graph_objects as go
from plotly.subplots import make_subplots


def build_matrices_figure(matrices, t, desc, n_cols=5):
    # create figure for matrices to display
    names = list(matrices.keys())
    n = len(matrices)
    n_rows = ceil(n / n_cols)
    fig = make_subplots(rows=n_rows, cols=n_cols,
                        vertical_spacing=0.05, horizontal_spacing=0.05)

    for i, name in enumerate(names):
        row = i // n_cols + 1
        col = i % n_cols + 1
        hm = go.Heatmap(
            name=name,
            x=desc['Name_Area'],
            y=desc['Name_Area'],
            z=(matrices[name][t]),
            zmin=-1, zmax=1,
            coloraxis='coloraxis',
            showscale=False,
            hovertemplate='Row: %{y}'+
                          '<br>Column: %{x}<br>'+
                          'Corr: %{z:.2f}',
        )
        fig.add_trace(hm, row=row, col=col)

    # set up the layout
    fig.update_layout(coloraxis=dict(colorscale='RdBu_r', cmin=-1, cmax=1), showlegend=False)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False, autorange='reversed')

    for i in range(1, n_rows + 1):
        for j in range(1, n_cols + 1):
            fig.update_xaxes(constrain='domain', row=i, col=j)
            fig.update_yaxes(scaleanchor=f'x{(i-1)*n_cols+j}', scaleratio=1,
                             constrain='domain', row=i, col=j)

    return fig


plotly_config = dict(displayModeBar='hover', displaylogo=False)
layout = [
    dbc.Row([
        dbc.Col(dbc.Label("Time"), width='auto'),
        dbc.Col(dcc.Slider(min=0, max=1, step=1, value=0, id='mtx-slider-time'))
    ]),
    dbc.Row(dcc.Graph(figure={}, id='mtx-graph', style={'height': '90vh'}, config=plotly_config)),
]


@callback(
    Output('mtx-graph', 'figure'),
    Output('mtx-slider-time', 'max'),
    Output('mtx-slider-time', 'marks'),
    Input('store-corr', 'data'),
    Input('mtx-slider-time', 'value'),
    State('store-desc', 'data'),
)
def update_figure(corr, slider_value, desc):
    if corr is None or len(corr) == 0:
        raise PreventUpdate

    max_slider_value = len(next(iter(corr.values()))) - 1
    marks_slider = {i: str(i) for i in range(0, max_slider_value + 1, max_slider_value//10)}
    return build_matrices_figure(corr, slider_value, desc), max_slider_value, marks_slider


# TODO add a callback to resize everything when graph is zoomed

# TODO add a callback to update the number of columns when the window is resized
