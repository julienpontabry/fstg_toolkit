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
from dash import html, dcc
import dash_bootstrap_components as dbc

from fstg_toolkit.app.core.datafilesdb import get_data_file_db
from fstg_toolkit.app.views.common import get_navbar


dash.register_page(__name__, path='/list')


def layout():
    db = get_data_file_db()
    return dbc.Container([
            get_navbar('/list'),
            html.H1("List of available datasets"),
            html.P("Click on a dataset token to open its dashboard in a new tab."),
            html.Ul([
                html.Li(dcc.Link(f"{token}", href=f'dashboard/{token}', target='new'))
                for token, _ in db.list()
            ])
        ],
        fluid='xxl')
