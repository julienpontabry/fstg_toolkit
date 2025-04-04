import plotly.io as pio
from dash import Dash, html, dcc

from app.views import data, subject

# use orsjon to make JSON 5-10x faster
pio.json.config.default_engine = 'orjson'

# app's definition
app = Dash(title="fSTView - An fMRI spatio-temporal data viewer", name="fSTView")

app.layout = html.Div([
    # app's layout
    dcc.Tabs([
        dcc.Tab(label="Data", children=data.layout),
        dcc.Tab(label="Subject view", children=subject.layout),
    ]),

    # app's storage cache
    dcc.Store(id='store-desc'),
    dcc.Store(id='store-corr')
])

if __name__ == '__main__':
    app.run(debug=True)
