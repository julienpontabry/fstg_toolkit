from dash.exceptions import PreventUpdate
from dash.dependencies import ALL
import dash_bootstrap_components as dbc
from dash_extensions.enrich import Input, Output, State, callback, dcc, html

from plotly import graph_objects as go

from fmri_st_graph.app.figures.subject import (
    build_subject_figure,
    generate_subject_display_props,
)
from fmri_st_graph.app.views.common import update_factor_controls, plotly_config


layout = [
    html.Div([], id='subject-factors-block'),
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
            children=[dcc.Graph(figure={}, id='st-graph', config=plotly_config)],
            type='circle', overlay_style={"visibility": "visible", "filter": "blur(2px)"}
        )
    )
]


@callback(
    Output('subject-factors-block', 'children'),
    Input('store-factors', 'data'),
    prevent_initial_call=True,
)
def update_subject_factor_controls(factors):
    return update_factor_controls('subject', factors, multi=False)


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
    Input({'type': 'subject-factor', 'index': ALL}, 'value'),
    State('subject-selection', 'value'),
)
def update_subjects(corr, factor_values, current_selection):
    if corr is None or len(corr) == 0 or len(factor_values) == 0:
        raise PreventUpdate

    # filter subjects based on selected factors
    filtered_corr_keys = filter(lambda k: all(f in factor_values for f in k[:-1]), corr.keys())
    filtered_corr_ids = list(map(lambda k: k[-1], filtered_corr_keys))

    # do not select a new subject in the filtered list if the old one is also in the filtered list
    selection = current_selection if current_selection in filtered_corr_ids else next(iter(filtered_corr_ids), None)

    return filtered_corr_ids, selection


@callback(
    Output('st-graph', 'figure'),
    Output('apply-button', 'disabled'),
    Input('apply-button', 'n_clicks'),
    Input('subject-selection', 'value'),
    Input('store-graphs', 'data'),
    State('regions-selection', 'value'),
    State({'type': 'subject-factor', 'index': ALL}, 'value'),
    prevent_initial_call=True
)
def update_graph(n_clicks, subject, graphs, regions, factor_values):
    if (n_clicks is not None and n_clicks <= 0) or graphs is None:
        raise PreventUpdate

    if subject is None or len(factor_values) == 0:
        raise PreventUpdate

    ids = tuple(factor_values + [subject])
    figure_props = generate_subject_display_props(graphs[ids], regions)

    return build_subject_figure(figure_props), True


@callback(
    Output('apply-button', 'disabled', allow_duplicate=True),
    Input('regions-selection', 'value'),
    prevent_initial_call=True
)
def enable_apply_button_at_selection_changed(regions):
    return regions is None or len(regions) == 0
