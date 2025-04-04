import plotly.io as pio
from dash import Dash, html, dcc

import stplot, data

# use orsjon to make JSON 5-10x faster
pio.json.config.default_engine = 'orjson'

# app's definition
app = Dash(title="FSTView - An fMRI spatio-temporal data viewer", name="FSTView")

app.layout = html.Div([
    # app's layout
    dcc.Tabs([
        dcc.Tab(label="Data", children=data.layout),
        dcc.Tab(label="Subject view", children=stplot.layout),
    ]),

    # app's storage cache
    dcc.Store(id='store-desc'),
])

if __name__ == '__main__':
    app.run(debug=True)
