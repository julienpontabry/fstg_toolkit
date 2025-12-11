# Copyright 2025 ICube (University of Strasbourg - CNRS)
# author: Julien PONTABRY (ICube)
#
# This software is a computer program whose purpose is to provide a toolkit
# to model, process and analyze the longitudinal reorganization of brain
# connectivity data, as functional MRI for instance.
#
# This software is governed by the CeCILL-B license under French law and
# abiding by the rules of distribution of free software. You can use,
# modify and/or redistribute the software under the terms of the CeCILL-B
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info".
#
# As a counterpart to the access to the source code and rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty and the software's author, the holder of the
# economic rights, and the successive licensors have only limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading, using, modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean that it is complicated to manipulate, and that also
# therefore means that it is reserved for developers and experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and, more generally, to use and operate it in the
# same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL-B license and that you accept its terms.

from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import dash_uploader as du
from dash import html, dcc, callback, Input, Output, State, set_props, callback_context

from fstg_toolkit.app.core.processing import SubmittedDataset, InvalidSubmittedDataset, get_dataset_processing_manager
from fstg_toolkit.app.views.common import get_navbar

dash.register_page(__name__, path='/submit')


name_input = html.Div([
        dbc.Label("Dataset name", html_for='dataset-name-input'),
        dbc.Input(type='text', id='dataset-name-input', placeholder='Enter dataset name'),
        dbc.FormText("A descriptive name for the dataset."),
    ], className='mb-3')

options_input = html.Div([
        dbc.Label("Options"),
        dbc.Checklist(options=[
                {'label': 'Include raw matrices', 'value': 'include_raw'},
                {'label': 'Compute metrics', 'value': 'compute_metrics'},
            ],
            value=['include_raw', 'compute_metrics'], id='dataset-options-input', inline=True, switch=True),
        dbc.FormText("Select what will be included in the dataset. Note that the spatio-temporal graph modeling is included anyway."),
    ], className='mb-3')

areas_upload = html.Div([
        dbc.Label("Areas description file (CSV)"),
        du.Upload(id='upload-areas-file', text="Drag and drop or click to select a file to upload.",
                  max_files=1, filetypes=['csv'], text_completed="Last upload: "),
        html.Div("", id='upload-areas-file-output'),
        dbc.FormText("The CSV file describing the areas and their regions they belong to."),
    ], className='mb-3')

matrices_upload = html.Div([
    dbc.Label("Matrices files (NPZ/NPY)"),
    du.Upload(id='upload-matrices-files', text="Drag and drop or click to select files to upload.",
              max_files=10, filetypes=['npy', 'npz'], text_completed="Last upload: "),
    html.Div("", id='upload-matrices-files-output'),
    dbc.FormText("One or multiple (max. 10) numpy pickle files containing the timeseries of correlation matrices."),
], className='mb-3')

form_buttons = html.Div([
        dbc.Button("Submit", id='submit-dataset-button', color='primary'),
        dbc.Button("Reset", id='reset-dataset-button', color='secondary', className='ms-2'),
    ], className='mb-3')


# FIXME make upload ids different at each submission; using same ids causes pending jobs to fail
layout = dbc.Container([
        get_navbar('/submit'),

        # form's layout
        html.H1("Submit a new dataset"),
        html.P("Use the form below to submit a new dataset to the fSTG-View application."),
        html.P("The dataset must contain at least:"),
        html.Ul([
            html.Li("a CSV file describing the areas and their regions they belong to;"),
            html.Li("one or more numpy pickle files (NPZ or NPY) containing the timeseries of correlation matrices.")
        ]),
        html.Hr(),
        dbc.Form([name_input, options_input, areas_upload, matrices_upload, form_buttons]),
        dbc.Alert(id='dataset-form-alert', children="", dismissable=True, fade=True, is_open=False),

        # storage to keep track of uploaded files
        dcc.Store(id='store-uploaded-areas-file', storage_type='memory'),
        dcc.Store(id='store-last-uploaded-areas-file', storage_type='memory'),
        dcc.Store(id='store-last-uploaded-matrices-files', storage_type='memory'),
        dcc.Store(id='store-uploaded-matrices-files', storage_type='memory')
    ],
    fluid='xxl')


def __make_file_list(files: list[str], prefix: str) -> html.Ul:
    return html.Ul([
        html.Li([Path(f).name, " ", html.A("", className='bi bi-trash',
                                           id={'type': f'{prefix}-remove-uploaded-file', 'index': f})])
        for f in files
    ])


@callback(
    Output('upload-areas-file-output', 'children'),
    Output('store-uploaded-areas-file', 'data'),
    Input('store-last-uploaded-areas-file', 'data'),
    Input({'type': 'areas-remove-uploaded-file', 'index': dash.ALL}, 'n_clicks'),
    State('store-uploaded-areas-file', 'data'),
    prevent_initial_callbacks=True
)
def update_uploaded_areas_file(last_uploaded_file, n_clicks, uploaded_file):
    ctx = callback_context

    if ctx.triggered_id == 'store-last-uploaded-areas-file':
        if not last_uploaded_file:
            return "", dash.no_update

        if uploaded_file is not None:   # remove previous file from disk
            Path(uploaded_file).unlink()

        return __make_file_list([last_uploaded_file], 'areas'), last_uploaded_file
    else:
        if not n_clicks:
            return dash.no_update, dash.no_update
        else:
            # clear the only file and clear the display
            Path(uploaded_file).unlink()
            return "", None


@callback(
    Output('upload-matrices-files-output', 'children'),
    Output('store-uploaded-matrices-files', 'data'),
    Input('store-last-uploaded-matrices-files', 'data'),
    Input({'type': 'matrices-remove-uploaded-file', 'index': dash.ALL}, 'n_clicks'),
    State('store-uploaded-matrices-files', 'data'),
    prevent_initial_callbacks=True
)
def update_uploaded_matrices_files(last_uploaded_files, n_clicks, uploaded_files):
    ctx = callback_context

    if ctx.triggered_id == 'store-last-uploaded-matrices-files':
        if not last_uploaded_files:
            return "", dash.no_update

        if not uploaded_files:
            all_uploaded_files = last_uploaded_files
        else:
            all_uploaded_files = uploaded_files + [
                f for f in last_uploaded_files if f not in uploaded_files
            ]

        return __make_file_list(all_uploaded_files, 'matrices'), all_uploaded_files
    else:
        if not n_clicks:
            return dash.no_update, dash.no_update
        else:
            # clear the requested file and update the display
            to_del = ctx.triggered_id['index']
            Path(to_del).unlink()
            all_uploaded_files = [f for f in uploaded_files if f != to_del]
            return __make_file_list(all_uploaded_files, 'matrices'), all_uploaded_files


@callback(
    Output('dataset-name-input', 'value'),
    Output('dataset-options-input', 'value'),
    Input('reset-dataset-button', 'n_clicks'),
    State('store-uploaded-areas-file', 'data'),
    State('store-uploaded-matrices-files', 'data'),
    prevent_initial_callbacks=True
)
def reset_dataset_form(_, areas_uploaded_file, matrices_uploaded_files):
    if areas_uploaded_file is not None:
        Path(areas_uploaded_file).unlink()
        set_props('store-last-uploaded-areas-file', {'data': None})
        set_props('store-uploaded-areas-file', {'data': None})

    if matrices_uploaded_files is not None and len(matrices_uploaded_files) > 0:
        for f in matrices_uploaded_files:
            Path(f).unlink()
        set_props('store-last-uploaded-matrices-files', {'data': None})
        set_props('store-uploaded-matrices-files', {'data': None})

    return "", ['include_raw', 'compute_metrics']


@callback(
    Input('submit-dataset-button', 'n_clicks'),
    State('dataset-name-input', 'value'),
    State('dataset-options-input', 'value'),
    State('store-uploaded-areas-file', 'data'),
    State('store-uploaded-matrices-files', 'data'),
    prevent_initial_callbacks=True
)
def submit_dataset_form(_, name, options, areas_uploaded_file, matrices_uploaded_files):
    try:
        # There is a minimal validation of the dataset at initialization.
        dataset = SubmittedDataset(
            name=name,
            include_raw='include_raw' in options,
            compute_metrics='compute_metrics' in options,
            areas_file=Path(areas_uploaded_file) if areas_uploaded_file else None,
            matrices_files=[Path(f) for f in matrices_uploaded_files] if matrices_uploaded_files else None)

        # Submit the dataset (no live check that the submission is going well;
        # user must monitor the progress on the list page).
        manager = get_dataset_processing_manager()
        manager.submit(dataset)
        for comp in ('dataset-name-input', 'upload-areas-file', 'upload-matrices-files',
                     'submit-dataset-button', 'reset-dataset-button'):
            set_props(comp, {'disabled': True})

        set_props('dataset-form-alert',{
            'children': ["The dataset has been submitted for processing. You can monitor the progress on the ",
                         html.A("list page", href='/list', className='alert-link'),
                         "."],
            'is_open': True,
            'color': 'success'})
    except InvalidSubmittedDataset as ex:
        set_props('dataset-form-alert', {'children': str(ex), 'is_open': True, 'color': 'danger'})
