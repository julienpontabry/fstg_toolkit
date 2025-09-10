from typing import Type

import pandas as pd
import plotly.express as px
from plotly import graph_objects as go


def __check_object_types(series: pd.DataFrame, dtype: Type):
    return series.apply(lambda e: isinstance(e, dtype)).all()


def build_metrics_plot(metrics: pd.DataFrame, selection: str):
    selected = metrics[selection]

    if isinstance(selected, pd.Series):
        return build_scalar_comparison_plot(selected)
    elif isinstance(selected, pd.DataFrame) and selected.columns.nlevels == 1:
        return build_distribution_comparison_plot(selected)
    else:
        return {}


def build_scalar_comparison_plot(metric: pd.Series):
    return px.violin(metric.reset_index(), x='factor2', color='factor1',
                     y=metric.name, box=True, points='all')


def build_distribution_comparison_plot(metrics: pd.DataFrame):
    factor = 'factor1'

    percentages = metrics.divide(metrics.sum(axis='columns'), axis='index') * 100
    values = percentages.groupby(factor).mean().T

    fig = go.Figure(data=[
        go.Bar(name=idx, x=values.columns, y=values.loc[idx].values)
        for idx in values.index
    ])

    fig.update_layout(
        barmode='stack',
        yaxis_type='log',
        xaxis_title=factor,
        yaxis_title="Percentage",
    )

    return fig
