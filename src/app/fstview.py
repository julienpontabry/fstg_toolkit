import plotly.io as pio
from app.views import data, matrices, subject
from dash_extensions.enrich import DashProxy, ServersideOutputTransform, dcc, html

# use orsjon to make JSON 5-10x faster
pio.json.config.default_engine = 'orjson'

# app's definition
app = DashProxy(title="fSTView - An fMRI spatio-temporal data viewer", name="fSTView",
                transforms=[ServersideOutputTransform()])

app.layout = html.Div([
    # app's layout
    dcc.Tabs([
        dcc.Tab(label="Data", children=data.layout),
        dcc.Tab(label="Raw data view", children=matrices.layout),
        dcc.Tab(label="Subject view", children=subject.layout),
    ]),

    # app's storage cache
    dcc.Store(id='store-desc'),
    dcc.Store(id='store-corr')
])

if __name__ == '__main__':
    app.run(debug=True)
