from dash.exceptions import PreventUpdate
from dash.dependencies import ALL
import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, dcc, html, clientside_callback, ClientsideFunction

from fmri_st_graph.app.figures.subject import build_subject_figure, generate_subject_display_props
from fmri_st_graph.app.views.common import update_factor_controls, plotly_config
from ..core.io import GraphsDataset


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
    ),

    dcc.Store(id='store-spatial-connections', storage_type='memory'),
]


@callback(
    Output('subject-factors-block', 'children'),
    Output('regions-selection', 'options'),
    Output('regions-selection', 'value'),
    Input('store-dataset', 'data'),
    prevent_initial_call=True,
)
def dataset_changed(store_dataset):
    if store_dataset is None:
        raise PreventUpdate

    dataset = GraphsDataset.deserialize(store_dataset)

    # update the layout of the factors' controls
    factor_controls_layout = update_factor_controls('subject', dataset.factors, multi=False)

    # update the selectable regions
    regions = dataset.areas_desc.sort_values("Name_Region")["Name_Region"].unique().tolist()

    return factor_controls_layout, regions, regions


@callback(
Output('subject-selection', 'options'),
    Output('subject-selection', 'value'),
    Input({'type': 'subject-factor', 'index': ALL}, 'value'),
    State('store-dataset', 'data'),
    State('subject-selection', 'value'),
    prevent_initial_call=True
)
def factors_changed(factor_values, store_dataset, current_selection):
    if store_dataset is None or factor_values is None:
        raise PreventUpdate

    # filter subjects based on selected factors
    ids = [tuple(record.values()) for record in store_dataset['subjects']]
    filtered_ids = filter(lambda k: all(f in factor_values for f in k[:-2]), ids)
    filtered_ids = list(map(lambda k: k[-2], filtered_ids))

    # do not select a new subject in the filtered list if the old one is also in the filtered list
    selection = current_selection if current_selection in filtered_ids else next(iter(filtered_ids), None)

    return filtered_ids, selection


@callback(
    Output('st-graph', 'figure'),
    Output('store-spatial-connections', 'data'),
    Output('apply-button', 'disabled'),
    Input('apply-button', 'n_clicks'),
    Input('subject-selection', 'value'),
    State('regions-selection', 'value'),
    State({'type': 'subject-factor', 'index': ALL}, 'value'),
    State('store-dataset', 'data'),
    prevent_initial_call=True
)
def selection_changed(n_clicks, subject, regions, factor_values, store_dataset):
    if (n_clicks is not None and n_clicks <= 0) or store_dataset is None:
        raise PreventUpdate

    if subject is None or len(factor_values) == 0:
        raise PreventUpdate

    # check if the graph is in the dataset
    ids = tuple(factor_values + [subject])
    dataset = GraphsDataset.deserialize(store_dataset)

    if ids not in dataset:
        raise PreventUpdate

    # loads the dataset and create figure properties from the loaded graph
    graph = dataset[ids]
    figure_props = generate_subject_display_props(graph, regions)

    return build_subject_figure(figure_props), figure_props['spatial_connections'], True


@callback(
    Output('apply-button', 'disabled', allow_duplicate=True),
    Input('regions-selection', 'value'),
    prevent_initial_call=True
)
def regions_selection_changed(regions):
    return regions is None or len(regions) == 0


clientside_callback(
    ClientsideFunction(
        namespace='clientside',
        function_name='subject_node_hover',
    ),
    Output('st-graph', 'style'), # NOTE this is a workaround to ensure the clientside callback is registered
    Input('st-graph', 'hoverData'),
    State('store-spatial-connections', 'data'),
    prevent_initial_call=True
)
