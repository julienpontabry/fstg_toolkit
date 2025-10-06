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
from dash import dcc
import dash_bootstrap_components as dbc


plotly_config = dict(displayModeBar='hover', displaylogo=False)


def update_factor_controls(prefix: str, factors: list[set[str]], multi: bool = True) -> list[dbc.Row]:
    if factors is None or len(factors) == 0:
        raise PreventUpdate

    controls = []

    for i, factor in enumerate(factors):
        factor_values = sorted(list(factor))

        if len(factor_values) > 0:
            value = factor_values if multi else factor_values[0]
            controls.append(dbc.Row([
                    dbc.Col(dbc.Label(f"Factor {i+1}"), width='auto'),
                    dbc.Col(dcc.Dropdown(options=factor_values, value=value,
                                         id={'type': f'{prefix}-factor', 'index': i},
                                         multi=multi, clearable=multi))
                ])
            )

    return controls
