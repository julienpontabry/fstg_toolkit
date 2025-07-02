from dash.exceptions import PreventUpdate
from dash.dependencies import ALL
from dash_extensions.enrich import Input, Output, State, callback, dcc, html
import dash_bootstrap_components as dbc

from app.figures.matrices import build_matrices_figure, break_width_to_cols
from app.views.common import update_factor_controls


plotly_config = dict(displayModeBar='hover', displaylogo=False)
layout = [
    html.Div([], id='mtx-factors-block'),
    dbc.Row([
        dbc.Col(dbc.Label("Time"), width='auto'),
        dbc.Col(dcc.Slider(min=0, max=1, step=1, value=0, id='mtx-slider-time'))
    ]),
    dbc.Row(dcc.Graph(figure={}, id='mtx-graph', config=plotly_config)),
]


@callback(
    Output('mtx-factors-block', 'children'),
    Input('store-factors', 'data'),
    prevent_initial_call=True,
)
def update_mtx_factor_controls(factors):
    return update_factor_controls('mtx', factors, multi=True)


@callback(
    Output('mtx-graph', 'figure'),
    Output('mtx-slider-time', 'max'),
    Output('mtx-slider-time', 'marks'),
    Input('store-corr', 'data'),
    Input('mtx-slider-time', 'value'),
    Input({'type': 'mtx-factor', 'index': ALL}, 'value'),
    Input('store-break-width', 'data'),
    State('store-desc', 'data'),
)
def update_figure(corr, slider_value, factor_values, break_width, desc):
    if corr is None or len(corr) == 0:
        raise PreventUpdate

    # filter the matrices based on the selected factors (if any)
    defined_factor_values = [factor_value for factor_value in factor_values
                             if factor_value is not None and len(factor_value) > 0]
    corr = {key: matrix for key, matrix in corr.items()
            if all(any(value in key for value in factor_value)
                   for factor_value in defined_factor_values)}

    # set the time slider properties
    max_slider_value = len(next(iter(corr.values()))) - 1
    marks_slider = {i: str(i) for i in range(0, max_slider_value + 1, max_slider_value//10)}

    # update the number of columns depending on the breakpoints width
    n_cols = break_width_to_cols(break_width['name'])

    # create figure
    return build_matrices_figure(corr, slider_value, desc, n_cols=n_cols), max_slider_value, marks_slider
