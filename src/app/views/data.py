import base64
import io

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from dash.exceptions import PreventUpdate
import dash_daq as daq
from dash_extensions.enrich import (
    Input,
    Output,
    Serverside,
    State,
    callback,
    dash_table,
    dcc,
    html,
    callback_context as ctx
)
from dash_extensions.logging import set_props
from dash.dependencies import ALL

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
                dcc.Upload(children=html.Div(["Drag and drop or ",
                                              html.A("select a description of regions/areas (.csv)")]),
                           multiple=False, id='upload-description', accept='.csv',
                           className='upload', className_active='upload-active'),
                ],
                type='circle', overlay_style={"visibility":"visible", "filter": "blur(2px)"}),
            dash_table.DataTable(columns=desc_columns, page_size=12, id='desc-table')
        ]),
        dbc.Col([
            html.H2("Correlation matrices"),
            dcc.Loading([
                    dcc.Upload(children=html.Div(["Drag and drop or ",
                                                  html.A("select correlation matrices files (.npy/.npz)")]),
                               multiple=True, id='upload-correlation', accept='.npy,.npz,.zip',
                               className='upload', className_active='upload-active'),
                    html.Div(id='uploaded-corr-files-list'),
                    dash_table.DataTable(columns=corr_columns, page_size=12, id='corr-table'),
                ],
                type='circle', overlay_style={"visibility": "visible", "filter": "blur(2px)"})
        ])
    ]),
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

    desc = pd.read_csv(io.StringIO(decoded.decode('utf-8')), index_col='Id_Area')
    return Serverside(desc)


@callback(
    Output('desc-table', 'data'),
    Input('store-desc', 'data'),
)
def populate_desc_table(desc):
    if desc is None or len(desc) == 0:
        raise PreventUpdate

    return desc.reset_index().to_dict('records')


# FIXME put that in an update_corr methods along with removing to avoid chained callbacks
@callback(
    Output('store-corr', 'data'),
    Input('upload-correlation', 'filename'),
    Input('upload-correlation', 'contents'),
    prevent_initial_call=True,
)
def upload_corr(filenames, contents):
    if contents is None:
        raise PreventUpdate

    if len(filenames) > 0 and all('npy' not in filename and
                                  'npz' not in filename and
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


# FIXME put that in an update_corr methods along with uploading to avoid chained callbacks
@callback(
    Output("upload-correlation", "filename"),
    Output("upload-correlation", "contents"),
    Input({"type": "remove-corr-file", "index": ALL}, "n_clicks"),
    State('upload-correlation', 'filename'),
    State('upload-correlation', 'contents'),
)
def remove_uploaded_file(remove_clicks, filenames, contents):
    # Find the first one that has been clicked
    for i, n in enumerate(remove_clicks):
        if n is not None and n > 0:
            idx = i
            break
    else:
        raise PreventUpdate

    # Remove the matching file
    new_filenames = list(filenames)
    new_contents = list(contents)
    del new_filenames[idx]
    del new_contents[idx]

    return new_filenames, new_contents


@callback(
    Output('uploaded-corr-files-list', 'children'),
    Input('upload-correlation', 'filename'),
    prevent_initial_call=True
)
def update_uploaded_corr_files_list(filenames):
    if not filenames:
        return None

    return dbc.ListGroup([
        dbc.ListGroupItem([filename,
                           dbc.Badge(html.I(className="bi bi-x-lg"), pill=True, color='danger',
                                     id={'type': 'remove-corr-file', 'index': i}, n_clicks=0,
                                     className="ms-1 file-remove")
                           ])
        for i, filename in enumerate(filenames)
    ])


@callback(
    Output('corr-table', 'data'),
    Input('store-corr', 'data')
)
def populate_corr_table(corr):
    if corr is None:
        raise PreventUpdate

    return [{'Subject': name} for name in corr.keys()]


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

    for i, (label, matrices) in enumerate(corr.items()):
        set_progress((str(i), str(n), f"Processing {label}...", 'visible'))
        try:
            graph[label] = spatio_temporal_graph_from_corr_matrices(
                matrices, desc, corr_thr=threshold, abs_thr=use_absolute)
        except Exception as ex:
            print(ex)
            errors.append(label)
        set_progress((str(i+1), str(n), f"Processing {label}...", 'visible'))

    set_progress((str(n), str(n), "Done.", 'visible'))

    if len(errors) > 0:
        set_props('message-toast', dict(
            is_open=True, header="Error", icon="danger", duration=None,
            children="An error occurred while processing the following subjects: " + ", ".join(errors)))

    return Serverside(graph)
