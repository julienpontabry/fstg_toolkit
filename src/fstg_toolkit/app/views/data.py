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
from dash.exceptions import PreventUpdate
from dash import Input, Output, State, callback, dash_table, dcc, html

from .common import (
    plotly_config,
    create_factors_options_controls,
    build_factors_options,
)
from ..core.io import GraphsDataset
from ..figures.data import areas_per_region_figure, subjects_per_factors_figure

desc_columns = [{'name': "Area id", 'id': 'Id_Area'},
                {'name': "Area name", 'id': 'Name_Area'},
                {'name': "Region name", 'id': 'Name_Region'}]
corr_columns = [{'name': "Subject", 'id': 'Subject'}]


layout = [
    dbc.Row(html.H2("Description of regions/areas")),
    dbc.Row([
        dbc.Col([
            dcc.Loading(
                children=dash_table.DataTable(
                    columns=desc_columns, page_size=15, id='desc-table',
                    sort_action='native', filter_action='native', style_as_list_view=True,
                    style_header={'fontWeight': 'bold', 'textAlign': 'center'}),
                type='circle', overlay_style={'visibility': 'visible', 'filter': 'blur(2px)'})
        ]),
        dbc.Col([
            dcc.Loading(
                dcc.Graph(figure={}, id='desc-count-plot',
                          config=dict(**plotly_config,
                                      modeBarButtonsToRemove=['select2d', 'lasso2d'])),
                type='circle', overlay_style={'visibility': 'visible', 'filter': 'blur(2px)'})
        ])
    ]),
    dbc.Row(html.H2("Subjects")),
    dbc.Row([
        dbc.Col([
            dcc.Loading(
                children=dash_table.DataTable(
                    columns=corr_columns, page_size=15, id='subjects-table',
                    sort_action='native', filter_action='native', style_as_list_view=True,
                    style_header={'fontWeight': 'bold', 'textAlign': 'center'}),
                type='circle', overlay_style={'visibility': 'visible', 'filter': 'blur(2px)'})
        ]),
        dbc.Col([
            dcc.Loading([
                    dbc.Row(dcc.Graph(
                        figure={}, id='subjects-dist-plot',
                        config=dict(**plotly_config, modeBarButtonsToRemove=['select2d', 'lasso2d']))),
                    dbc.Row(create_factors_options_controls('data-subjects-dist'))
                ],
                type='circle', overlay_style={'visibility': 'visible', 'filter': 'blur(2px)'}
            )
        ])
    ]),
]


@callback(
    Output('desc-table', 'data'),
    Output('desc-count-plot', 'figure'),
    Output('subjects-table', 'columns'),
    Output('subjects-table', 'data'),
    Output('data-subjects-dist-factors', 'options'),
    Output('data-subjects-dist-factors', 'value'),
    Input('store-dataset', 'data'),
    prevent_initial_call=True
)
def dataset_changed(store_dataset):
    if store_dataset is None:
        return PreventUpdate

    # update the columns of subjects table
    n_factors = len(store_dataset['factors'])
    columns = [{'name': f"Factor {i + 1}", 'id': f'Factor{i}'}
               for i in range(n_factors)]
    columns.append({'name': "Subject", 'id': 'Subject'})

    # compute plot for areas distribution
    areas_count_fig = areas_per_region_figure(store_dataset['areas_desc'])

    # update the factors selection for factors distribution plot
    factors, default_factors = build_factors_options(GraphsDataset.deserialize(store_dataset))

    return store_dataset['areas_desc'], areas_count_fig, columns, store_dataset['subjects'], \
        factors, default_factors

@callback(
    Output('subjects-dist-plot', 'figure'),
    Input('data-subjects-dist-factors', 'value'),
    State('store-dataset', 'data'),
    prevent_initial_call=True
)
def factors_selection_changed(factors, store_dataset):
    if store_dataset is None:
        raise PreventUpdate

    return subjects_per_factors_figure(store_dataset['subjects'])
