from dash.exceptions import PreventUpdate
from dash.dependencies import ALL
from dash_extensions.enrich import Input, Output, State, callback, dcc, html
import dash_bootstrap_components as dbc

from app.figures.matrices import build_matrices_figure


plotly_config = dict(displayModeBar='hover', displaylogo=False)
layout = [
    html.Div([], id='factors-block'),
    dbc.Row([
        dbc.Col(dbc.Label("Time"), width='auto'),
        dbc.Col(dcc.Slider(min=0, max=1, step=1, value=0, id='mtx-slider-time'))
    ]),
    dbc.Row(dcc.Graph(figure={}, id='mtx-graph', style={'height': '90vh'}, config=plotly_config)),
]


@callback(
    Output('factors-block', 'children'),
    Input('store-factors', 'data'),
    prevent_initial_call=True,
)
def update_factor_controls(factors):
    if factors is None or len(factors) == 0:
        raise PreventUpdate

    controls = []

    for i, factor in enumerate(factors):
        controls.append(dbc.Row([
                dbc.Col(dbc.Label(f"Factor {i+1}"), width='auto'),
                dbc.Col(dcc.Dropdown(list(factor), id={'type': 'mtx-factor', 'index': i}))
            ])
        )

    return controls


@callback(
    Output('mtx-graph', 'figure'),
    Output('mtx-slider-time', 'max'),
    Output('mtx-slider-time', 'marks'),
    Input('store-corr', 'data'),
    Input('mtx-slider-time', 'value'),
    Input({'type': 'mtx-factor', 'index': ALL}, 'value'),
    State('store-desc', 'data'),
)
def update_figure(corr, slider_value, factor_values, desc):
    if corr is None or len(corr) == 0:
        raise PreventUpdate

    # filter the matrices based on the selected factors (if any)
    defined_factor_values = [factor_value for factor_value in factor_values
                             if factor_value is not None]
    corr = {key: matrix for key, matrix in corr.items()
            if all(factor_value in key for factor_value in defined_factor_values)}

    # set the time slider properties
    max_slider_value = len(next(iter(corr.values()))) - 1
    marks_slider = {i: str(i) for i in range(0, max_slider_value + 1, max_slider_value//10)}

    # create figure
    return build_matrices_figure(corr, slider_value, desc), max_slider_value, marks_slider


# TODO add a callback to resize everything when graph is zoomed

# TODO add a callback to update the number of columns when the window is resized
