import plotly.io as pio
from dash import Dash, html

import stplot

# use orsjon to make JSON 5-10x faster
pio.json.config.default_engine = 'orjson'

app = Dash(title="FSTView - An fMRI spatio-temporal data viewer", name="FSTView")
app.layout = html.Div([
    stplot.layout,
])

if __name__ == '__main__':
    app.run(debug=True)
