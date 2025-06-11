from dash.exceptions import PreventUpdate
from dash_extensions.enrich import Input, Output, State, callback, dcc
import dash_bootstrap_components as dbc

from app.figures.matrices import build_matrices_figure


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
