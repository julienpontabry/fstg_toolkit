import base64
import io

import dash_bootstrap_components as dbc
import numpy as np
from dash.dependencies import ALL
from dash.exceptions import PreventUpdate
from dash_extensions.enrich import (
    Input,
    Output,
    Serverside,
    State,
    callback,
    dash_table,
    dcc,
    html
)

from fmri_st_graph.app.core.utils import split_factors_from_name

desc_columns = [{'name': "Area id", 'id': 'Id_Area'},
                {'name': "Area name", 'id': 'Name_Area'},
                {'name': "Region name", 'id': 'Name_Region'}]
corr_columns = [{'name': "Subject", 'id': 'Subject'}]

layout = [
    dbc.Row([
        dbc.Col([
            html.H2("Description of regions/areas"),
            dcc.Loading(
                children=dash_table.DataTable(columns=desc_columns, page_size=15, id='desc-table'),
                type='circle', overlay_style={'visibility': 'visible', 'filter': 'blur(2px)'})
        ]),
        dbc.Col([
            html.H2("Subjects"),
            dcc.Loading([
                    dcc.Upload(children=html.Div(["Drag and drop or ",
                                                  html.A("select correlation matrices files (.npy/.npz)",
                                                         className='upload-link')]),
                               multiple=True, id='upload-correlation', accept='.npy,.npz,.zip',
                               className='upload', className_active='upload-active'),
                    html.Div(id='uploaded-corr-files-list'),
                    dash_table.DataTable(columns=corr_columns, page_size=15, id='corr-table'),
                ],
                type='circle', overlay_style={"visibility": "visible", "filter": "blur(2px)"})
        ])
    ]),
]


@callback(
    Output('desc-table', 'data'),
    Output('corr-table', 'columns'),
    Output('corr-table', 'data'),
    Input('store-dataset', 'data'),
    prevent_initial_call=True
)
def dataset_changed(store_dataset):
    if store_dataset is None:
        return PreventUpdate

    # update the columns of subjects table
    nb_cols = len(store_dataset['subjects'][0])
    columns = [{'name': f"Factor {i + 1}", 'id': f'Factor{i}'}
               for i in range(nb_cols - 1)]  # -1 to account for index and subject column
    columns.append({'name': "Subject", 'id': 'Subject'})

    return store_dataset['areas_desc'], columns, store_dataset['subjects']


# FIXME put that in an update_corr methods along with removing to avoid chained callbacks
@callback(
    Output('store-factors', 'data'),
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

    # load data
    files = {filename: content for filename, content in zip(filenames, contents)}
    corr = {}

    for _, content in files.items():
        content_type, content_string = content.split(',')
        decoded = base64.b64decode(content_string)
        data = np.load(io.BytesIO(decoded))

        for name, matrices in data.items():
            corr[name] = matrices

    # extract factors and ids from names
    factors, ids = split_factors_from_name(corr.keys())

    return Serverside(factors), Serverside({ident: corr[name] for name, ident in zip(corr.keys(), ids)})


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


# @callback(
#     Output('corr-table', 'columns'),
#     Output('corr-table', 'data'),
#     Input('store-corr', 'data'),
#     State('store-factors', 'data')
# )
# def populate_corr_table(corr, factors):
#     if corr is None:
#         raise PreventUpdate
#
#     # update columns
#     columns = [{'name': f"Factor {i+1}", 'id': f'Factor{i}'}
#                for i in range(len(factors))]
#     columns.append({'name': "Subject", 'id': 'Subject'})
#
#     # update contents
#     contents = []
#     for ids in corr.keys():
#         desc = {'Subject': ids[-1]}
#         for i, factor in enumerate(ids[:-1]):
#             desc[f'Factor{i}'] = factor
#         contents.append(desc)
#
#     return columns, contents
