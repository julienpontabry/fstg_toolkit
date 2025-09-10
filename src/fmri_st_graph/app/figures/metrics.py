import pandas as pd
import plotly.express as px
from plotly import graph_objects as go


def build_metrics_plot(metric: pd.DataFrame | pd.Series, factors: list[str]):
    if isinstance(metric, pd.Series):
        return build_scalar_comparison_plot(metric, factors)
    elif isinstance(metric, pd.DataFrame) and metric.columns.nlevels == 1:
        return build_distribution_comparison_plot(metric, factors)
    else:
        return {}


def build_scalar_comparison_plot(metric: pd.Series, factors: list[str]):
    params = {param: factor for param, factor in zip(('x', 'color'), factors)}
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
        go.Bar(name=idx, x=labels, y=values.loc[idx])
        for idx in values.index
    ])

    fig.update_layout(
        barmode='stack',
        yaxis_type='log',
        xaxis_title=x_label,
        yaxis_title="Percentage",
    )

    return fig
