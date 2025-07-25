import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from dash import Input, Output, callback, dash_table, dcc, html


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
            dcc.Loading(
                children=dash_table.DataTable(columns=corr_columns, page_size=15, id='corr-table'),
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
    n_factors = len(store_dataset['factors'])
    columns = [{'name': f"Factor {i + 1}", 'id': f'Factor{i}'}
               for i in range(n_factors)]
    columns.append({'name': "Subject", 'id': 'Subject'})

    return store_dataset['areas_desc'], columns, store_dataset['subjects']
