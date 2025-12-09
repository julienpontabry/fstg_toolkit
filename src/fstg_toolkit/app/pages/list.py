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

from datetime import datetime
from typing import Optional

import dash
from dash import html, callback, Input, Output, State, MATCH
import dash_bootstrap_components as dbc

from fstg_toolkit.app.core.processing import get_dataset_processing_manager
from fstg_toolkit.app.views.common import get_navbar
from fstg_toolkit.app.core.processing import ProcessingJobStatus


dash.register_page(__name__, path='/list')


def __status2color(job_status: ProcessingJobStatus) -> str:
    match job_status:
        case ProcessingJobStatus.PENDING|ProcessingJobStatus.RUNNING:
            return "warning"
        case ProcessingJobStatus.COMPLETED:
            return "success"
        case ProcessingJobStatus.FAILED:
            return "danger"


def __make_status_badge(status: Optional[ProcessingJobStatus]) -> dbc.Badge:
    if status is not None:
        label = status.value
        color = __status2color(status)
    else:
        label = "Unknown"
        color = "info"

    return dbc.Badge(label, color=color, className="me-1")


def __format_time_ago(timestamp: datetime) -> str:
    diff = datetime.now() - timestamp

    if diff.days > 0:
        return f"{diff.days} day{'s' if diff.days > 1 else ''} ago"

    hours = diff.seconds // 3600
    if hours > 0:
        return f"{hours} hour{'s' if hours > 1 else ''} ago"

    minutes = (diff.seconds % 3600) // 60
    if minutes > 0:
        return f"{minutes} minute{'s' if minutes > 1 else ''} ago"

    if diff.seconds > 0:
        return f"{diff.seconds} second{'s' if diff.seconds > 1 else ''} ago"

    return "just now"


def layout():
    manager = get_dataset_processing_manager()
    return dbc.Container(
        [
            get_navbar("/list"),
            html.H1("List of last submitted datasets"),
            html.P("Click on a dataset to open its dashboard in a new tab. "
                   "Click on the left arrow to expand the dataset card and see additional informations."),
            html.Hr(),
            # TODO currently url token of result is missing
            dbc.Container(dbc.Stack([
                dbc.Card([
                    dbc.CardHeader(
                        dbc.Row([
                            dbc.Col([
                                html.I(className='bi bi-caret-right-fill', id={'type': 'dataset-arrow', 'index': i},
                                       style={'font-size': '1.5rem', 'color': 'gray', 'cursor': 'pointer'})
                            ], width=1, align='center'),
                            dbc.Col([
                                html.H4(result.dataset.name, className='card-title'),
                                html.Small(f"Submitted {__format_time_ago(result.submitted_at)}", className='text-muted'),
                            ]),
                            dbc.Col([
                                html.Div(__make_status_badge(result.job_status), className='text-end')
                            ], width=3, align='center')
                        ])
                    ),
                    dbc.Collapse(
                        dbc.CardBody(html.Ul([
                            html.Li([html.B("Include raw data: "), 'Yes' if result.dataset.include_raw else 'No']),
                            html.Li([html.B("Compute metrics: "), 'Yes' if result.dataset.compute_metrics else 'No']),
                            html.Li([html.B("Areas description file: "), html.I(result.dataset.areas_file.name)]),
                            html.Li([
                                html.B("Matrices files: "),
                                html.Ul([
                                    html.Li(html.I(str(mat_path.name)))
                                    for mat_path in result.dataset.matrices_files
                                ])
                            ]),
                            *([html.Hr(), html.P(result.error if result.error else "", className="text-danger text-center")]
                            if result.error else [])
                        ]), className='card-text'), is_open=False, id={'type': 'dataset-sup-info', 'index': i}),
                ])
                for i, result in enumerate(manager.list())
            ], gap=3), style={'max-width': '800px', 'align': 'center'})
        ], fluid='xxl')

@callback(
    Output({'type': 'dataset-sup-info', 'index': MATCH}, 'is_open'),
    Output({'type': 'dataset-arrow', 'index': MATCH}, 'className'),
    Input({'type': 'dataset-arrow', 'index': MATCH}, 'n_clicks'),
    State({'type': 'dataset-sup-info', 'index': MATCH}, 'is_open'),
    prevent_initial_callbacks=True
)
def toggle_collapse(n_clicks, is_open):
    if not n_clicks or is_open:
        return False, 'bi bi-caret-right-fill'
    else:
        return True, 'bi bi-caret-down-fill'
