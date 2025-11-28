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

import dash
import dash_bootstrap_components as dbc
from dash import Input,Output, State, callback, dcc, html
from dash_breakpoints import WindowBreakpoints

from fstg_toolkit.app.views import metrics, data, subject, matrices
from fstg_toolkit.app.core.datafilesdb import get_data_file_db
from fstg_toolkit.app.core.io import GraphsDataset


dash.register_page(__name__, path_template='/dashboard/<token>')


def dashboard_layout(serialized_dataset, matrices_disabled, metrics_disabled):
    return dbc.Container(
        children=[
            # app's layout
            dbc.Tabs([
                    dbc.Tab(label="Dataset", id='tab-data', tab_id='tab-data', children=data.layout),
                    dbc.Tab(label="Raw data", id='tab-matrices', tab_id='tab-matrices', children=matrices.layout, disabled=matrices_disabled),
                    dbc.Tab(label="Subject", id='tab-subject', tab_id='tab-subject', children=subject.layout, disabled=False),
                    dbc.Tab(label="Metrics", id='tab-population', tab_id='tab-population', children=metrics.layout, disabled=metrics_disabled),
                ],
                id='tabs'),

            # app's storage cache
            dcc.Store(id='store-dataset', storage_type='memory', data=serialized_dataset),
            dcc.Store(id='store-break-width', storage_type='memory'),

            # setup event on window's width breakpoints
            # FIXME this should be on all pages
            WindowBreakpoints(
                id='window-width-break',
                widthBreakpointThresholdsPx=[576, 768, 992, 1200, 1400],
                widthBreakpointNames=['xsm', 'sm', 'md', 'lg', 'xl', 'xxl'],
            ),

            # message display as toasts
            # FIXME this should be on all pages
            dbc.Toast('', id='message-toast', header='', icon='primary', duration=4_000, is_open=False,
                      dismissable=True, style={'position': 'fixed', 'bottom': 10, 'right': 10, 'width': 350}),
        ],
        fluid='xxl')


def layout(token=None):
    db = get_data_file_db()

    if filepath := db.get(token):
        dataset = GraphsDataset.from_filepath(filepath)
        return dashboard_layout(
            serialized_dataset=dataset.serialize(),
            matrices_disabled=not dataset.has_matrices(),
            metrics_disabled=not dataset.has_metrics())
    else:
        return dbc.Container(
            children=[
                html.H1("Unable to show dashboard"),
                html.P(f"The token '{token}' is invalid or the dataset does not exist."),
            ],
            fluid='xxl',
        )


@callback(
    Output('store-break-width', 'data'),
    Input('window-width-break', 'widthBreakpoint'),
    State('window-width-break', 'width')
)
def store_current_break_width(breakpoint_name, breakpoint_width):
    return {'name': breakpoint_name, 'width': breakpoint_width}
