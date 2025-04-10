import dash_bootstrap_components as dbc
import diskcache
import plotly.io as pio
from app.views import data, matrices, subject
from dash import DiskcacheManager
from dash_extensions.enrich import (
    DashProxy,
    Input,
    ServersideOutputTransform,
    callback,
    dcc,
    set_props,
)

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

# app's definition
app = DashProxy(title="fSTView - An fMRI spatio-temporal data viewer", name="fSTView",
                transforms=[ServersideOutputTransform()],
                external_stylesheets=[dbc.themes.BOOTSTRAP],
                background_callback_manager=background_callback_manager,
                on_error=callback_error)

app.layout = dbc.Container(
    children=[
        # app's layout
        dbc.Tabs([
            dbc.Tab(label="Data", id='tab-data', tab_id='tab-data', children=data.layout),
            dbc.Tab(label="Raw data view", id='tab-matrices', children=matrices.layout, disabled=True),
            dbc.Tab(label="Subject view", id='tab-subject', children=subject.layout, disabled=True),
        ], id='tabs'),

        # app's storage cache
        dcc.Store(id='store-desc', storage_type='session'),
        dcc.Store(id='store-corr', storage_type='session'),
        dcc.Store(id='store-graph', storage_type='session'),

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
    Input('store-corr', 'data')
)
def set_tab_enabled(desc, corr):
    if all(e is not None for e in (desc, corr)):
        for tab in ('tab-matrices', 'tab-subject'):
            set_props(tab, dict(disabled=False))
        set_props('tabs', {'active_tab': 'tab-data'})


if __name__ == '__main__':
    app.run(debug=True)
