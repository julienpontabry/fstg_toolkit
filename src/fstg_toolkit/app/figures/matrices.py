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

from math import ceil

from plotly import graph_objects as go
from plotly.subplots import make_subplots


def build_matrices_figure(matrices, t, desc, n_cols=3, vs_ratio=0.1, hs_ratio=0.2, matrix_height=300):
    # prepare figure
    ids = list(matrices.keys())
    n = len(matrices)
    n_rows = ceil(n / n_cols)
    fig = make_subplots(rows=n_rows, cols=n_cols, shared_xaxes='all', shared_yaxes='all',
                        horizontal_spacing=hs_ratio / n_cols, vertical_spacing=vs_ratio / n_rows,
                        subplot_titles=[f"{ident[-1]} {'/'.join(ident[:-1])}"
                                        for ident in __keep_nonunique_factors(ids)])

    # create heatmaps for matrices
    for i, ident in enumerate(ids):
        row = i // n_cols + 1
        col = i % n_cols + 1
        hm = go.Heatmap(
            name='/'.join(ident),
            x=desc['Name_Area'],
            y=desc['Name_Area'],
            z=(matrices[ident][t]),
            zmin=-1, zmax=1,
            coloraxis='coloraxis',
            hovertemplate='Row: %{y}'+
                          '<br>Column: %{x}<br>'+
                          'Corr: %{z:.2f}',
        )
        fig.add_trace(hm, row=row, col=col)

    # set up the layout
    fig.update_layout(coloraxis={
            'colorscale': 'RdBu_r', 'cmin': -1, 'cmax': 1,
            'colorbar': {
                'orientation': 'h', 'yanchor': 'bottom',
                'title': {'text': "Correlation", 'side': 'top'}
            }
        }, showlegend=False)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False, autorange='reversed')

    # fix layout
    fig_height = (1 + vs_ratio / (1 - vs_ratio)) * matrix_height * n_rows
    fig.update_layout(height=fig_height)

    for i in range(1, n_rows + 1):
        for j in range(1, n_cols + 1):
            fig.update_xaxes(constrain='domain', row=i, col=j)
            fig.update_yaxes(scaleanchor=f'x{(i-1)*n_cols+j}', scaleratio=1,
                             constrain='domain', row=i, col=j)

    return fig


def break_width_to_cols(break_name: str) -> int:
    if break_name == 'xsm':
        return 1
    elif break_name == 'sm':
        return 2
    elif break_name == 'md':
        return 3
    elif break_name == 'lg':
        return 4
    elif break_name == 'xl':
        return 5
    else:
        return 6


def __keep_nonunique_factors(ids: list[tuple[any, ...]]) -> list[tuple[any, ...]]:
    factors_vals = list(zip(*ids))
    tmp = [f_vals for f_vals in factors_vals[:-1] if len(set(f_vals)) > 1]
    tmp.append(factors_vals[-1])
    return list(zip(*tmp))
