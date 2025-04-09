import plotly.io as pio
from app.views import data, matrices, subject
from dash_extensions.enrich import DashProxy, ServersideOutputTransform, dcc, html
import dash_bootstrap_components as dbc

# use orsjon to make JSON 5-10x faster
pio.json.config.default_engine = 'orjson'

# app's definition
app = DashProxy(title="fSTView - An fMRI spatio-temporal data viewer", name="fSTView",
                transforms=[ServersideOutputTransform()],
                external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container(
    children=[
        # app's layout
        dbc.Tabs([
            dbc.Tab(label="Data", children=data.layout),
            dbc.Tab(label="Raw data view", children=matrices.layout),
            dbc.Tab(label="Subject view", children=subject.layout, disabled=False),
        ]),

        # app's storage cache
        dcc.Store(id='store-desc', storage_type='session'),
        dcc.Store(id='store-corr', storage_type='session'),
    ],
    fluid='xxl'
)

if __name__ == '__main__':
    app.run(debug=True)
