import traceback as tb
from pathlib import Path

import dash_bootstrap_components as dbc
import diskcache
import plotly.io as pio
from dash import DiskcacheManager
from dash_extensions.enrich import (
    DashProxy,
    Input,
    Output,
    State,
    ServersideOutputTransform,
    callback,
    dcc,
    set_props,
    Serverside
)
from dash_breakpoints import WindowBreakpoints

from .views import population, data, subject, matrices
from .core.datafilesdb import get_data_file_db
from ..io import load_spatio_temporal_graphs

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
app = DashProxy(title="fSTView - An fMRI spatio-temporal data viewer", name="fSTView",
                transforms=[ServersideOutputTransform()],
                external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP],
                assets_folder=str(Path(__file__).parent / 'assets'),
                background_callback_manager=background_callback_manager,
                on_error=callback_error)

app.layout = dbc.Container(
    children=[
        # browser address
        dcc.Location(id='url'),

        # TODO add a fullscreen app loading page when loading data

        # app's layout
        dbc.Tabs([
            dbc.Tab(label="Data", id='tab-data', tab_id='tab-data', children=data.layout),
            dbc.Tab(label="Data view", id='tab-matrices', tab_id='tab-matrices',
                    children=matrices.layout, disabled=True),
            dbc.Tab(label="Subject view", id='tab-subject', tab_id='tab-subject',
                    children=subject.layout, disabled=True),
            dbc.Tab(label="Population view", id='tab-population', tab_id='tab-population',
                    children=population.layout, disabled=True),
        ], id='tabs'),

        # app's storage cache
        dcc.Store(id='store-desc', storage_type='session'),
        dcc.Store(id='store-factors', storage_type='session'),
        dcc.Store(id='store-corr', storage_type='session'),
        dcc.Store(id='store-graphs', storage_type='session'),
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
    Input('store-desc', 'data'),
    Input('store-corr', 'data'),
    State('tabs', 'active_tab')
)
def set_tab_enabled_after_data(desc, corr, active_tab):
    if all(e is not None for e in (desc, corr)):
        set_props('tab-matrices', dict(disabled=False))
        set_props('tab-model', dict(disabled=False))
        set_props('tabs', {'active_tab': active_tab})


@callback(
    Input('store-graphs', 'data'),
    State('tabs', 'active_tab')
)
def set_tab_enabled_after_model(graphs, active_tab):
    if graphs is not None:
        set_props('tab-subject', dict(disabled=False))
        set_props('tab-population', dict(disabled=False))
        set_props('tabs', {'active_tab': active_tab})


@callback(
    Output('store-break-width', 'data'),
    Input('window-width-break', 'widthBreakpoint'),
    State('window-width-break', 'width')
)
def store_current_break_width(breakpoint_name, breakpoint_width):
    return {'name': breakpoint_name, 'width': breakpoint_width}


@callback(
    Output('store-desc', 'data', allow_duplicate=True),
    Output('store-graphs', 'data', allow_duplicate=True),
    Input('url', 'pathname'),
    prevent_initial_call=True
)
def data_file_has_changed(pathname):
    # get file path from token in pathname
    db = get_data_file_db()
    filepath = db.get(pathname[1:])

    # store the areas and graphs
    graphs = load_spatio_temporal_graphs(filepath)
    areas_desc = graphs[next(iter(graphs))].areas

    return Serverside(areas_desc), Serverside(graphs)


if __name__ == '__main__':
    app.run(debug=True)
