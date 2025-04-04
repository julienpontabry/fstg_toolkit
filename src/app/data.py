import base64
import io

import numpy as np
import pandas as pd

from dash import html, dcc, dash_table, callback, Input, Output


desc_columns = [{'name': "Area id", 'id': 'Id_Area'},
                {'name': "Area name", 'id': 'Name_Area'},
                {'name': "Region name", 'id': 'Name_Region'}]

layout = [
    html.H2("Description of regions/areas"),
    dcc.Upload(children=["Drag and drop or select a description of regions/areas (.csv)"],
               multiple=False, id='upload-description'),
    dash_table.DataTable(columns=desc_columns, page_size=12, id='desc-table'),
    html.H2("Correlation matrices"),
    dcc.Upload(children=["Drag and drop or select correlation matrices files (.npy/.npz)"],
               multiple=True, id='upload-correlation'),
    dcc.Loading([html.Div(id='loading-correlation')], type='circle',
                overlay_style={"visibility":"visible", "filter": "blur(2px)"})
]


@callback(
Output('store-desc', 'data'),
    Output('desc-table', 'data'),
    Input('upload-description', 'filename'),
    Input('upload-description', 'contents'),
)
def upload_description(filename, contents):
    if contents is None:
        return None, None

    if 'csv' not in filename:
        return None, None

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    try:
        desc = pd.read_csv(io.StringIO(decoded.decode('utf-8')), index_col='Id_Area')
        return desc.to_json(), desc.reset_index().to_dict('records')
    except Exception as e:  # TODO display the error with a toaster or something?
        print(f"Error reading CSV: {e}")
        return None, None


@callback(
    Output('store-corr', 'data'),
    Output('loading-correlation', 'children'),
    Input('upload-correlation', 'filename'),
    Input('upload-correlation', 'contents'),
)
def upload_corr(filenames, contents):
    if contents is None:
        return None, ""

    if all('npy' not in filename and \
           'npz' not in filename and \
           'zip' not in filename
           for filename in filenames):
        return None, "Filename extension not supported!"

    files = {filename: content for filename, content in zip(filenames, contents)}
    corr = {}

    for filename, content in files.items():
        content_type, content_string = content.split(',')
        decoded = base64.b64decode(content_string)
        data = np.load(io.BytesIO(decoded))

        for name, matrices in data.items():
            corr[name] = matrices

    return corr, "Data loaded successfully!"
