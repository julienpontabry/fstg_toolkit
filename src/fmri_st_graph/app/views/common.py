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
