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
from dash import html
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
        return f"{diff.days}{'s' if diff.days > 1 else ''} ago"

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
    return dbc.Container([
            get_navbar('/list'),
            html.H1("List of last submitted datasets"),
            html.P("Click on a dataset to open its dashboard in a new tab."),
            html.Hr(),
            # TODO currently url token of result is missing
            dbc.CardGroup([
                dbc.Card(
                    dbc.CardBody([
                        html.H4(result.dataset.name, className='card-title'),
                        html.H6(__make_status_badge(result.job_status), className='card-subtitle'),
                        html.P("Some information", className='card-text'),
                        html.Small(f"Submitted {__format_time_ago(result.submitted_at)}", className="card-text text-muted"),
                        html.Br(),
                        dbc.CardLink("Open dashboard")
                    ])
                )
                for result in manager.list()
            ])
        ],
        fluid='xxl')
