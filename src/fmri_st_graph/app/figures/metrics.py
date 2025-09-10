from typing import Type

import pandas as pd
import plotly.express as px


def __check_object_types(series: pd.DataFrame, dtype: Type):
    return series.apply(lambda e: isinstance(e, dtype)).all()


def build_metrics_plot(metrics: pd.DataFrame, selection: str):
    selected = metrics[selection]

    if isinstance(selected, pd.Series):
        return build_scalar_comparison_plot(selected)
    elif __check_object_types(metrics[selection], list):
        return build_distribution_comparison_plot(metrics, selection)
    else:
        return {}


def build_scalar_comparison_plot(metric: pd.Series):
    return px.violin(metric.reset_index(), x='factor2', color='factor1',
                     y=metric.name, box=True, points='all')


def build_distribution_comparison_plot(metrics: pd.DataFrame, selection: str):
    return px.bar(metrics, x="factor1", y=selection)
