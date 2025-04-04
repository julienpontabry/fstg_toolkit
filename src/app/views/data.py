import base64
import io

import numpy as np
import pandas as pd

from dash import html, dcc, dash_table, callback, Input, Output
from dash.exceptions import PreventUpdate

desc_columns = [{'name': "Area id", 'id': 'Id_Area'},
                {'name': "Area name", 'id': 'Name_Area'},
                {'name': "Region name", 'id': 'Name_Region'}]
corr_columns = [{'name': "Subject", 'id': 'Subject'}]

layout = [
    html.H2("Description of regions/areas"),
    dcc.Loading([
        dcc.Upload(children=["Drag and drop or select a description of regions/areas (.csv)"],
                   multiple=False, id='upload-description'),
    ],
    type='circle', overlay_style={"visibility":"visible", "filter": "blur(2px)"}),
    dash_table.DataTable(columns=desc_columns, page_size=12, id='desc-table'),
    html.H2("Correlation matrices"),
    dcc.Loading([
        dcc.Upload(children=["Drag and drop or select correlation matrices files (.npy/.npz)"],
                   multiple=True, id='upload-correlation'),
        dash_table.DataTable(columns=corr_columns, page_size=12, id='corr-table'),
    ],
    type='circle', overlay_style={"visibility":"visible", "filter": "blur(2px)"})
]


@callback(
Output('store-desc', 'data'),
    Output('desc-table', 'data'),
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
        return desc.to_json(), desc.reset_index().to_dict('records')
    except Exception as e:  # TODO display the error with a toaster or something?
        print(f"Error reading CSV: {e}")
        raise PreventUpdate


@callback(
    Output('store-corr', 'data'),
    Output('corr-table', 'data'),
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

    return corr, [{'Subject': name} for name in corr.keys()]
