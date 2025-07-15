import base64
import io

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
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
            dcc.Loading([
                dcc.Upload(children=html.Div(["Drag and drop or ",
                                              html.A("select a description of regions/areas (.csv)",
                                                     className='upload-link')]),
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
                                                  html.A("select correlation matrices files (.npy/.npz)",
                                                         className='upload-link')]),
                               multiple=True, id='upload-correlation', accept='.npy,.npz,.zip',
                               className='upload', className_active='upload-active'),
                    html.Div(id='uploaded-corr-files-list'),
                    dash_table.DataTable(columns=corr_columns, page_size=12, id='corr-table'),
                ],
                type='circle', overlay_style={"visibility": "visible", "filter": "blur(2px)"})
        ])
    ]),
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


@callback(
    Output('corr-table', 'columns'),
    Output('corr-table', 'data'),
    Input('store-corr', 'data'),
    State('store-factors', 'data')
)
def populate_corr_table(corr, factors):
    if corr is None:
        raise PreventUpdate

    # update columns
    columns = [{'name': f"Factor {i+1}", 'id': f'Factor{i}'}
               for i in range(len(factors))]
    columns.append({'name': "Subject", 'id': 'Subject'})

    # update contents
    contents = []
    for ids in corr.keys():
        desc = {'Subject': ids[-1]}
        for i, factor in enumerate(ids[:-1]):
            desc[f'Factor{i}'] = factor
        contents.append(desc)

    return columns, contents
