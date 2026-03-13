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
from dash import html

from fstg_toolkit.app.views.common import get_navbar

dash.register_page(__name__, path='/')


layout = dbc.Container([
        get_navbar(),
        html.H1("Welcome to fSTG-View serving", className="mb-4"),
        html.P(
            "fSTG-View processes fMRI correlation data into spatio-temporal graphs and provides "
            "interactive dashboards for exploring the results. Follow the steps below to get started."
        ),
        html.Hr(),
        html.H1("Getting started", className="mb-4"),

        # Step 1 — Submit
        dbc.Row([
            dbc.Col(html.H3([
                dbc.Badge("1", color="primary", className="me-2"),
                html.A("Submit a dataset", href="/submit"),
            ]), width=12),
            dbc.Col(html.P([
                "Go to the ", html.A("Submit", href="/submit"), " page to upload your data. You will need:"
            ]), width=12),
            dbc.Col(html.Ul([
                html.Li("An areas description file (CSV) listing brain areas and their regions."),
                html.Li("One or more correlation-matrix files (NPZ or NPY, up to 10)."),
            ]), width=12),
            dbc.Col(html.P([
                "You can also choose optional processing steps: ",
                html.B("Include raw matrices"), " (keeps the original matrices in the output), ",
                html.B("Compute metrics"), " (graph-level spatial and temporal metrics), and ",
                html.B("Compute frequent patterns"), " (frequent subgraph mining via SPMiner). "
                "After clicking ", html.B("Submit"), ", processing starts in the background."
            ]), width=12),
        ], className="mb-4"),

        # Step 2 — Monitor
        dbc.Row([
            dbc.Col(html.H3([
                dbc.Badge("2", color="primary", className="me-2"),
                html.A("Monitor processing", href="/list"),
            ]), width=12),
            dbc.Col(html.P([
                "The ", html.A("List", href="/list"),
                " page shows all submitted datasets and their current status:"
            ]), width=12),
            dbc.Col(html.Ul([
                html.Li([dbc.Badge("Pending", color="info", className="me-1"), "— waiting in the processing queue."]),
                html.Li([dbc.Badge("Running", color="warning", className="me-1"), "— currently being processed."]),
                html.Li([dbc.Badge("Completed", color="success", className="me-1"), "— processing finished successfully."]),
                html.Li([dbc.Badge("Failed", color="danger", className="me-1"), "— an error occurred (expand the card for details)."]),
            ]), width=12),
            dbc.Col(html.P(
                "Click the arrow on the left of a dataset card to expand it and see the submitted files, "
                "selected options, and any error message."
            ), width=12),
        ], className="mb-4"),

        # Step 3 — Dashboard
        dbc.Row([
            dbc.Col(html.H3([
                dbc.Badge("3", color="primary", className="me-2"),
                "Explore the dashboard",
            ]), width=12),
            dbc.Col(html.P([
                "Once a dataset is completed, two icons appear next to it in the List page:"
            ]), width=12),
            dbc.Col(html.Ul([
                html.Li([
                    html.I(className="bi bi-clipboard-data me-1"),
                    html.B("Dashboard"), " — opens the interactive dashboard in a new tab.",
                ]),
                html.Li([
                    html.I(className="bi bi-download me-1"),
                    html.B("Download"), " — downloads the processed dataset as a ZIP archive.",
                ]),
            ]), width=12),
            dbc.Col(html.P("The dashboard is organised into tabs:"), width=12),
            dbc.Col(html.Ul([
                html.Li([html.B("Dataset"), " — overview of graph-level properties across the full dataset."]),
                html.Li([html.B("Raw data"), " — browse the original correlation matrices (if included)."]),
                html.Li([html.B("Subject"), " — per-subject spatio-temporal graph visualisation and inspection."]),
                html.Li([html.B("Metrics"), " — spatial and temporal graph metrics across the population (if computed)."]),
                html.Li([html.B("Frequent Patterns"), " — frequent subgraph patterns discovered by SPMiner (if computed)."]),
            ]), width=12),
        ], className="mb-4"),
    ],
    fluid='xxl')
