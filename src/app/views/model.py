
import dash_bootstrap_components as dbc
import dash_daq as daq
from dash.exceptions import PreventUpdate
from dash_extensions.enrich import html, Input, Output, State, callback, Serverside
from dash_extensions.logging import set_props

from fmri_st_graph import spatio_temporal_graph_from_corr_matrices


layout = [
    dbc.Row([
        html.H2("Spatio-temporal graph model"),
        dbc.Form(
            dbc.Row([
                dbc.Col(dbc.Row([
                    dbc.Col(dbc.Label("Correlation threshold", width='auto'), width='auto'),
                    dbc.Col(dbc.Input(id='model-threshold', type='number', min=0, max=1, step=0.01, value=0.4)),
                ]), class_name='mx-2'),
                dbc.Col(dbc.Row([
                    dbc.Col(dbc.Label("Use absolute correlation", width='auto'), width='auto'),
                    dbc.Col(daq.BooleanSwitch(id='model-use-absolute', on=True), width='auto'),
                ]), class_name='mx-2'),
                dbc.Col([
                        dbc.Button("Process", color='primary', id='model-process-button'),
                        # NOTE: cancelling is not compatible with dash extensions yet...
                        # dbc.Button("Cancel", color='danger', id='model-cancel-button', disabled=True),
                    ],
                    width='auto')
            ])
        ),
    ]),
    dbc.Row(dbc.Label(id='model-process-label')),
    dbc.Row(dbc.Progress(id='model-process-progress', class_name='invisible'))
]


@callback(
    Output('store-graphs', 'data'),
    Input('model-process-button', 'n_clicks'),
    State('model-threshold', 'value'),
    State('model-use-absolute', 'on'),
    State('store-desc', 'data'),
    State('store-corr', 'data'),
    prevent_initial_call=True,
    background=True,
    running=[
        (Output('model-threshold', 'disabled'), True, False),
        (Output('model-use-absolute', 'disabled'), True, False),
        (Output('model-process-button', 'disabled'), True, False),
        # NOTE: cancelling is not compatible with dash extensions yet...
        # (Output('model-cancel-button', 'disabled'), False, True)
    ],
    progress=[
        Output('model-process-progress', 'value'),
        Output('model-process-progress', 'max'),
        Output('model-process-label', 'children'),
        Output('model-process-progress', 'class_name')
    ],
    # NOTE: cancelling is not compatible with dash extensions yet...
    # cancel=[
    #     Input('model-cancel-button', 'n_clicks')
    # ]
)
def compute_model(set_progress, n_clicks, threshold, use_absolute, desc, corr):
    if n_clicks <= 0 or any(e is None for e in (desc, corr)):
        raise PreventUpdate

    n = len(corr)
    set_progress((str(0), str(n), f"Processing...", 'visible'))
    graph = dict()
    errors = []

    for i, (ids, matrices) in enumerate(corr.items()):
        set_progress((str(i), str(n), f"Processing {'/'.join(ids)}...", 'visible'))
        try:
            graph[ids] = spatio_temporal_graph_from_corr_matrices(
                matrices, desc, corr_thr=threshold, abs_thr=use_absolute)
        except Exception as ex:
            print(ex)
            errors.append(ids)
        set_progress((str(i+1), str(n), f"Processing {'/'.join(ids)}...", 'visible'))

    set_progress((str(n), str(n), "Done.", 'visible'))

    if len(errors) > 0:
        set_props('message-toast', dict(
            is_open=True, header="Error", icon="danger", duration=None,
            children="An error occurred while processing the following subjects: " + ", ".join(errors)))

    return Serverside(graph)
