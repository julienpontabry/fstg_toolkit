from typing import Type

import pandas as pd
import plotly.express as px


def __check_object_types(series: pd.DataFrame, dtype: Type):
    return series.apply(lambda e: isinstance(e, dtype)).all()


def build_metrics_plot(metrics: pd.DataFrame, selection: str):
    if metrics[selection].dtype == float:
        return build_scalar_comparison_plot(metrics, selection)
    elif __check_object_types(metrics[selection], list):
        return build_distribution_comparison_plot(metrics, selection)
    else:
        return {}


def build_scalar_comparison_plot(metrics: pd.DataFrame, selection: str):
    return px.violin(metrics, x="factor2", color="factor1",
                     y=selection, box=True, points="all")


def build_distribution_comparison_plot(metrics: pd.DataFrame, selection: str):
    return px.bar(metrics, x="factor1", y=selection)
