from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from dash_extensions.enrich import Input, Output, State, callback, dcc

from app.figures.subject import build_subject_figure


layout = [
    dbc.Row(
        dcc.Dropdown([], clearable=False, id='subject-selection')
    ),
    dbc.Row([
        dbc.Col(dcc.Dropdown([], multi=True, placeholder="Select regions...", id='regions-selection'), width=11),
        dbc.Col(dbc.Button("Apply", color='secondary', id='apply-button'),
                className='d-grid gap-2 d-md-block', align='center')
    ], className='g-0'),
    dbc.Row(
        dcc.Loading(
            children=[dcc.Graph(figure={}, id='st-graph')],
            type='circle', overlay_style={"visibility": "visible", "filter": "blur(2px)"}
        )
    )
]


@callback(
    Output('regions-selection', 'options'),
    Output('regions-selection', 'value'),
    Input('store-desc', 'data'),
)
def update_regions(desc):
    if desc is None:
        raise PreventUpdate

    regions = desc.sort_values("Name_Region")["Name_Region"].unique().tolist()
    return regions, regions


@callback(
Output('subject-selection', 'options'),
    Output('subject-selection', 'value'),
    Input('store-corr', 'data'),
)
def update_subjects(corr):
    if corr is None or len(corr) == 0:
        raise PreventUpdate

    return ['/'.join(ids) for ids in corr.keys()], '/'.join(next(iter(corr.keys())))


@callback(
    Output('st-graph', 'figure'),
    Output('apply-button', 'disabled'),
    Input('apply-button', 'n_clicks'),
    Input('subject-selection', 'value'),
    Input('store-graphs', 'data'),
    State('regions-selection', 'value'),
    prevent_initial_call=True
)
def update_graph(n_clicks, ids, graphs, regions):
    if (n_clicks is not None and n_clicks <= 0) or graphs is None:
        raise PreventUpdate

    return build_subject_figure(graphs[tuple(ids.split('/'))], ids, regions), True


@callback(
    Output('apply-button', 'disabled', allow_duplicate=True),
    Input('regions-selection', 'value'),
    prevent_initial_call=True
)
def enable_apply_button_at_selection_changed(regions):
    return regions is None or len(regions) == 0
