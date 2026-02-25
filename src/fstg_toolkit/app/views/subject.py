# Copyright 2025 ICube (University of Strasbourg - CNRS)
# author: Julien PONTABRY (ICube)
#
# This software is a computer program whose purpose is to provide a toolkit
# to model, process and analyze the longitudinal reorganization of brain
# connectivity data, as functional MRI for instance.
#
# This software is governed by the CeCILL-B license under French law and
# abiding by the rules of distribution of free software. You can use,
# modify and/or redistribute the software under the terms of the CeCILL-B
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info".
#
# As a counterpart to the access to the source code and rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty and the software's author, the holder of the
# economic rights, and the successive licensors have only limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading, using, modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean that it is complicated to manipulate, and that also
# therefore means that it is reserved for developers and experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and, more generally, to use and operate it in the
# same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL-B license and that you accept its terms.

import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, dcc, html, clientside_callback, ClientsideFunction
from dash.dependencies import ALL
from dash.exceptions import PreventUpdate

from .common import update_factor_controls, plotly_config
from ..core.io import GraphsDataset
from ..figures.subject import build_subject_figure, generate_temporal_graph_props, build_spatial_figure, \
    generate_spatial_graph_props

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
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Spatial view", id='modal-sp-title')),
        dbc.ModalBody(dcc.Graph(figure={}, id='sp-graph',
                                config=dict(**plotly_config,
                                            modeBarButtonsToRemove=['zoom', 'pan', 'zoomIn', 'zoomOut',
                                                                    'autoScale', 'resetScale']))),
    ], id='modal-sp-graph', size='xl', centered=True, is_open=False),

    dcc.Store(id='store-spatial-connections', storage_type='memory'),
]


@callback(
    Output('subject-factors-block', 'children'),
    Output('regions-selection', 'options'),
    Output('regions-selection', 'value'),
    Input('store-dataset', 'data')
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
    # records contains "factors" ... "subject" "graph filename" ["matrix filename"]
    # we are interested in the elements until the subject
    n = len(store_dataset['factors'])
    ids = [tuple(record.values()) for record in store_dataset['subjects']]
    filtered_ids = filter(lambda k: all(f in factor_values for f in k[:n]), ids)
    filtered_ids = sorted([k[n] for k in filtered_ids])

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
    graph = dataset.get_graph(ids)
    figure_props = generate_temporal_graph_props(graph, regions)

    areas = dataset.areas_desc['Name_Area']
    return build_subject_figure(figure_props, areas), figure_props['spatial_connections'], True


@callback(
    Output('apply-button', 'disabled', allow_duplicate=True),
    Input('regions-selection', 'value'),
    prevent_initial_call=True
)
def regions_selection_changed(regions):
    return regions is None or len(regions) == 0


clientside_callback(
    ClientsideFunction(namespace='clientside', function_name='subject_node_hover'),
    Output('st-graph', 'style'), # NOTE this is a workaround to ensure the clientside callback is registered
    Input('st-graph', 'hoverData'),
    State('store-spatial-connections', 'data'),
    prevent_initial_call=True
)


# NOTE workaround to remove the hover elements on the graph when the mouse gets out of the figure
clientside_callback(
    ClientsideFunction(namespace='clientside', function_name='subject_clear_out'),
    Output('st-graph', 'id'),
    Input('st-graph', 'id')
)


@callback(
    Output('sp-graph', 'figure'),
    Output('modal-sp-graph', 'is_open'),
    Output('modal-sp-title', 'children'),
    Input('st-graph', 'clickData'),
    State('store-dataset', 'data'),
    State('regions-selection', 'value'),
    State({'type': 'subject-factor', 'index': ALL}, 'value'),
    State('subject-selection', 'value'),
    prevent_initial_call=True,
)
def graph_clicked(click_data, store_dataset, regions, factor_values, subject):
    if store_dataset is None or click_data is None:
        raise PreventUpdate

    # get time point value
    t = click_data['points'][0]['x']

    # check if the graph is in the dataset
    ids = tuple(factor_values + [subject])
    dataset = GraphsDataset.deserialize(store_dataset)

    if ids not in dataset:
        raise PreventUpdate

    # loads the dataset and create figure properties from the loaded graph
    graph = dataset.get_graph(ids)
    figure_props = generate_spatial_graph_props(graph.sub(t=t), dataset.areas_desc, regions)

    return build_spatial_figure(figure_props), True, f"Spatial view at t={t}"
