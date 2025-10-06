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

import pandas as pd
import plotly.express as px
from plotly import graph_objects as go


def build_metrics_plot(metric: pd.DataFrame | pd.Series, factors: list[str]):
    if isinstance(metric, pd.Series):
        if 'Time' in metric.index.names:
            return build_longitudinal_scalar_comparison_plot(metric, factors)
        else:
            return build_scalar_comparison_plot(metric, factors)
    elif isinstance(metric, pd.DataFrame) and metric.columns.nlevels == 1:
        return build_distribution_comparison_plot(metric, factors)
    else:
        return {}


def build_longitudinal_scalar_comparison_plot(metric: pd.Series, factors: list[str]):
    group = metric.groupby(['Time'] + factors)
    df = pd.DataFrame(group.mean())
    df['std'] = group.std()
    params = {param: factor for param, factor in zip(('color', 'facet_row', 'facet_col'), factors)}
    return px.line(df.reset_index(), x='Time', y=metric.name, **params, height=800)


def build_scalar_comparison_plot(metric: pd.Series, factors: list[str]):
    params = {param: factor for param, factor in zip(('x', 'color', 'facet_row', 'facet_col'), factors)}
    return px.violin(metric.reset_index(), y=metric.name, box=True, points='all', **params)


def build_distribution_comparison_plot(metric: pd.DataFrame, factors: list[str]):
    percentages = metric.divide(metric.sum(axis='columns'), axis='index') * 100

    if len(factors) > 0:
        values = percentages.groupby(factors[0]).mean().T
        labels = list(values.columns)
        x_label = factors[0]
    else:
        values = pd.DataFrame(percentages.mean())
        labels = ["all"]
        x_label = ""

    fig = go.Figure(data=[
        go.Bar(name=idx, x=labels, y=values.loc[idx], hovertemplate="%{y:.2f}%")
        for idx in values.index
    ])

    fig.update_layout(
        barmode='stack',
        yaxis_type='log',
        xaxis_title=x_label,
        yaxis_title="Percentage",
    )

    return fig
