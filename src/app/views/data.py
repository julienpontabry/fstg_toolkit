import base64
import io

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from dash.exceptions import PreventUpdate
from dash_extensions.enrich import (
    Input,
    Output,
    State,
    Serverside,
    callback,
    dash_table,
    dcc,
    html
)

from fmri_st_graph import spatio_temporal_graph_from_corr_matrices

desc_columns = [{'name': "Area id", 'id': 'Id_Area'},
                {'name': "Area name", 'id': 'Name_Area'},
                {'name': "Region name", 'id': 'Name_Region'}]
corr_columns = [{'name': "Subject", 'id': 'Subject'}]

layout = [
    dbc.Row([
        dbc.Col([
            html.H2("Description of regions/areas"),
            dcc.Loading([
                dcc.Upload(children=["Drag and drop or select a description of regions/areas (.csv)"],
                           multiple=False, id='upload-description'),
                ],
                type='circle', overlay_style={"visibility":"visible", "filter": "blur(2px)"}),
            dash_table.DataTable(columns=desc_columns, page_size=12, id='desc-table')
        ]),
        dbc.Col([
            html.H2("Correlation matrices"),
            dcc.Loading([
                    dcc.Upload(children=["Drag and drop or select correlation matrices files (.npy/.npz)"],
                               multiple=True, id='upload-correlation'),
                    dash_table.DataTable(columns=corr_columns, page_size=12, id='corr-table'),
                ],
                type='circle', overlay_style={"visibility": "visible", "filter": "blur(2px)"})
        ])
    ]),
    dbc.Row([
        html.H2("Spatio-temporal graph model"),
        dbc.Form(
            dbc.Row([
                dbc.Label("Correlation threshold", width='auto'),
                dbc.Col(dbc.Input(type='number', min=0, max=1, step=0.01, value=0.4)),
                dbc.Col(dbc.Checkbox(label="Use absolute correlation", value=True)),
                dbc.Col(dbc.Button("Process", color='primary', id='model-process-button'), width='auto')
            ])
        ),
        dcc.Interval(id='model-process-monitor', disabled=True),
        dbc.Progress(id='model-process-progress')
    ])
]


@callback(
Output('store-desc', 'data'),
    Input('upload-description', 'filename'),
    Input('upload-description', 'contents'),
)
def upload_description(filename, contents):
    if contents is None:
        raise PreventUpdate

    if 'csv' not in filename:
        raise PreventUpdate

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    try:
        desc = pd.read_csv(io.StringIO(decoded.decode('utf-8')), index_col='Id_Area')
        return Serverside(desc)
    except Exception as e:  # TODO display the error with a toaster or something?
        print(f"Error reading CSV: {e}")
        raise PreventUpdate


@callback(
    Output('desc-table', 'data'),
    Input('store-desc', 'data'),
)
def populate_desc_table(desc):
    if desc is None or len(desc) == 0:
        raise PreventUpdate

    return desc.reset_index().to_dict('records')


@callback(
    Output('store-corr', 'data'),
    Input('upload-correlation', 'filename'),
    Input('upload-correlation', 'contents'),
)
def upload_corr(filenames, contents):
    if contents is None:
        raise PreventUpdate

    if all('npy' not in filename and \
           'npz' not in filename and \
           'zip' not in filename
           for filename in filenames):
        raise PreventUpdate

    files = {filename: content for filename, content in zip(filenames, contents)}
    corr = {}

    for filename, content in files.items():
        content_type, content_string = content.split(',')
        decoded = base64.b64decode(content_string)
        data = np.load(io.BytesIO(decoded))

        for name, matrices in data.items():
            corr[name] = matrices

    return Serverside(corr)


@callback(
    Output('corr-table', 'data'),
    Input('store-corr', 'data')
)
def populate_corr_table(corr):
    if corr is None or len(corr) == 0:
        raise PreventUpdate

    return [{'Subject': name} for name in corr.keys()]


@callback(
    Input('model-process-button', 'n_clicks'),
    State('store-desc', 'data'),
    State('store-corr', 'data'),
    State('cache-model-progress', 'data'),
    prevent_initial_call=True,
    running=[
        (Output('model-process-button', 'disabled'), True, False),
        (Output('model-process-monitor', 'disabled'), False, True)
    ]
    # background=True  # TODO use a background manager
)
def compute_model(_, desc, corr, cache):
    # TODO get threshold and absolute correlation from the form
    if any(e is None for e in (desc, corr)):
        return

    cache['label'] = ""
    cache['progress'] = 0.
    n = len(corr)

    try:
        for i, (label, matrices) in enumerate(corr.items()):
            cache['label'] = label  # FIXME does not persist data to other callbacks
            graph = spatio_temporal_graph_from_corr_matrices(matrices, desc)
            cache['progress'] = 100 * (i+1) / n  # FIXME same here
            if i > 3:
                break
    except Exception as ex:
        print(ex)


@callback(
    Output('model-process-progress', 'value'),
    Output('model-process-progress', 'label'),
    Input('model-process-monitor', 'n_intervals'),
    State('cache-model-progress', 'data'),
    prevent_initial_call=True
)
def update_process_progress(n, cache):
    if cache is None:
        raise PreventUpdate

    print(n, cache)
    return cache['progress'], cache['label']
