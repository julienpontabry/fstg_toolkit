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
    Input('store-dataset', 'data')
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

    # deserialize the dataset and filter the matrices to load (if any)
    dataset = GraphsDataset.deserialize(store_dataset)

    if not dataset.has_matrices():
        raise PreventUpdate

    def_fac_vals = list(filter(lambda f: f is not None and len(f) > 0, factor_values))
    selected = sorted(filter(lambda ids: all(any(v in ids for v in f) for f in def_fac_vals),
                      dataset.subjects.index))

    # load the selected matrices
    corr = {ids: dataset.get_matrix(ids) for ids in selected}

    # set the time slider properties
    max_slider_value = len(next(iter(corr.values()))) - 1
    marks_slider = {i: str(i) for i in range(0, max_slider_value + 1, max_slider_value//10)}

    # update the number of columns depending on the breakpoints width and create figure
    n_cols = break_width_to_cols(break_width['name'])
    figure = build_matrices_figure(corr, slider_value, dataset.areas_desc, n_cols=n_cols)

    return figure, max_slider_value, marks_slider
