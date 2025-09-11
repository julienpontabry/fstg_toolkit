import traceback as tb
from pathlib import Path

import dash_bootstrap_components as dbc
import diskcache
import plotly.io as pio
from dash import DiskcacheManager, Dash, Input,Output, State, callback, dcc, set_props
from dash_breakpoints import WindowBreakpoints

from .views import metrics, data, subject, matrices
from .core.datafilesdb import get_data_file_db
from .core.io import GraphsDataset

# use orsjon to make JSON 5-10x faster
pio.json.config.default_engine = 'orjson'

# set up the cache
cache = diskcache.Cache("./cache")
background_callback_manager = DiskcacheManager(cache)

# handling of errors messages
def callback_error(err):
    set_props('message-toast', dict(
        is_open=True, header="Error", icon="danger", duration=None,
        children=str(err)))
    print(err)

    if err_tb := getattr(err, '__traceback__', None):
        tb.print_tb(err_tb)

# app's definition
app = Dash(title="fSTView - An fMRI spatio-temporal data viewer", name="fSTView",
           external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP],
           assets_folder=str(Path(__file__).parent / 'assets'),
           background_callback_manager=background_callback_manager,
           on_error=callback_error)

app.layout = dbc.Container(
    children=[
        # browser address
        dcc.Location(id='url'),

        # app's layout
        dbc.Tabs([
            dbc.Tab(label="Dataset", id='tab-data', tab_id='tab-data', children=data.layout),
            dbc.Tab(label="Raw data", id='tab-matrices', tab_id='tab-matrices',
                    children=matrices.layout, disabled=True),
            dbc.Tab(label="Subject", id='tab-subject', tab_id='tab-subject',
                    children=subject.layout, disabled=False),
            dbc.Tab(label="Metrics", id='tab-population', tab_id='tab-population',
                    children=metrics.layout, disabled=True),
        ], id='tabs'),

        # app's storage cache
        dcc.Store(id='store-dataset', storage_type='memory'),
        dcc.Store(id='store-break-width', storage_type='memory'),

        # setup event on window's width breakpoints
        WindowBreakpoints(
            id='window-width-break',
            widthBreakpointThresholdsPx=[576, 768, 992, 1200, 1400],
            widthBreakpointNames=['xsm', 'sm', 'md', 'lg', 'xl', 'xxl'],
        ),

        # message display as toasts
        dbc.Toast("",
                  id="message-toast", header="", icon="primary",
                  duration=4_000, is_open=False, dismissable=True,
                  style={'position': 'fixed', 'bottom': 10, 'right': 10, 'width': 350})
    ],
    fluid='xxl'
)


@callback(
    Output('store-break-width', 'data'),
    Input('window-width-break', 'widthBreakpoint'),
    State('window-width-break', 'width')
)
def store_current_break_width(breakpoint_name, breakpoint_width):
    return {'name': breakpoint_name, 'width': breakpoint_width}


@callback(
    Output('store-dataset', 'data'),
    Output('tab-matrices', 'disabled'),
    Output('tab-population', 'disabled'),
    Input('url', 'pathname'),
    prevent_initial_call=True
)
def pathname_changed(pathname):
    db = get_data_file_db()
    filepath = db.get(pathname[1:])  # remove first character, which is '/'

    # load lazily the dataset and enable tabs accordingly
    dataset = GraphsDataset.from_filepath(filepath)
    disable_matrices_view = not dataset.has_matrices()
    disable_metrics_view = not dataset.has_metrics()

    return dataset.serialize(), disable_matrices_view, disable_metrics_view
