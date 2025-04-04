import base64
import io

import pandas as pd

from dash import dcc, callback, Input, Output

layout = [
    dcc.Upload(children=["Drag and drop or select a description of regions/areas (.csv)"],
               multiple=False, id='upload-description'),
    dcc.Upload(children=["Drag and drop or select correlation matrices files (.npy/.npz)"],
               multiple=True, id='upload-correlation'),
]


@callback(
Output('store-desc', 'data'),
    Input('upload-description', 'filename'),
    Input('upload-description', 'contents'),
)
def upload_description(filename, contents):
    if contents is None:
        return None

    if 'csv' not in filename:
        return None

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    try:
        desc = pd.read_csv(io.StringIO(decoded.decode('utf-8')), index_col='Id_Area')
        return desc.to_json()
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None
