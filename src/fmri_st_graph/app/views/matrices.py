from dash.exceptions import PreventUpdate
from dash.dependencies import ALL
from dash import Input, Output, State, callback, dcc, html
import dash_bootstrap_components as dbc

from ..figures.matrices import build_matrices_figure, break_width_to_cols
from .common import update_factor_controls, plotly_config
from ..core.io import GraphsDataset


layout = [
    html.Div([], id='mtx-factors-block'),
    dbc.Row([
        dbc.Col(dbc.Label("Time"), width='auto'),
        dbc.Col(dcc.Slider(min=0, max=1, step=1, value=0, id='mtx-slider-time'))
    ]),
    dbc.Row(
        dcc.Loading(
            children=[dcc.Graph(figure={}, id='mtx-graph', config=plotly_config)],
            type='circle', overlay_style={"visibility": "visible", "filter": "blur(2px)"}
        )
    ),
]


@callback(
    Output('mtx-factors-block', 'children'),
    Input('store-dataset', 'data'),
    prevent_initial_call=True,
)
def dataset_changed(store_dataset):
    if store_dataset is None:
        raise PreventUpdate

    # update the layout of the factors' controls
    return update_factor_controls('mtx', store_dataset['factors'], multi=True)


@callback(
    Output('mtx-graph', 'figure'),
    Output('mtx-slider-time', 'max'),
    Output('mtx-slider-time', 'marks'),
    Input('mtx-slider-time', 'value'),
    Input({'type': 'mtx-factor', 'index': ALL}, 'value'),
    Input('store-break-width', 'data'),
    State('store-dataset', 'data'),
    prevent_initial_call=True
)
def selection_changed(slider_value, factor_values, break_width, store_dataset):
    if store_dataset is None:
        raise PreventUpdate

    # deserialize the dataset and filter the matrices to load
    dataset = GraphsDataset.deserialize(store_dataset)

    def_fac_vals = list(filter(lambda f: f is not None and len(f) > 0, factor_values))
    selected = filter(lambda ids: all(any(v in ids for v in f) for f in def_fac_vals),
                      dataset.subjects.index)

    # load the selected matrices
    corr = {ids: dataset.get_matrix(ids) for ids in selected}

    # set the time slider properties
    max_slider_value = len(next(iter(corr.values()))) - 1
    marks_slider = {i: str(i) for i in range(0, max_slider_value + 1, max_slider_value//10)}

    # update the number of columns depending on the breakpoints width and create figure
    n_cols = break_width_to_cols(break_width['name'])
    figure = build_matrices_figure(corr, slider_value, dataset.areas_desc, n_cols=n_cols)

    return figure, max_slider_value, marks_slider
