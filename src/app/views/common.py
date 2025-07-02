from dash.exceptions import PreventUpdate
from dash_extensions.enrich import dcc
import dash_bootstrap_components as dbc


def update_factor_controls(prefix: str, factors: list[set[str]], multi: bool =True):
    if factors is None or len(factors) == 0:
        raise PreventUpdate

    controls = []

    for i, factor in enumerate(factors):
        factor_values = list(factor)
        controls.append(dbc.Row([
                dbc.Col(dbc.Label(f"Factor {i+1}"), width='auto'),
                dbc.Col(dcc.Dropdown(options=factor_values, value=factor_values,
                                     multi=multi, id={'type': f'{prefix}-factor', 'index': i}))
            ])
        )

    return controls
