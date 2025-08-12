window.dash_clientside = Object.assign({}, window.dash_clientside, {
    clientside: {
        subject_node_hover: function(hoverData, storeHoverGraph) {
            // check hover data input
            if (!hoverData || !hoverData.points || hoverData.points.length === 0 || !storeHoverGraph) {
                return window.dash_clientside.no_update;
            }
        
            // check for a graph figure with specified id
            const graphDiv = document.getElementById('st-graph');
            const figure = graphDiv.querySelector('.js-plotly-plot');
        
            if (!figure || !figure.data) {
                return window.dash_clientside.no_update;
            }
            
            // check for any neighbouring coordinates associated with current point
            const point = hoverData.points[0];
            const x = point['x'];
            const y = point['y'];
            const props = storeHoverGraph[x][y]; // contains y coordinates and weights

            if (!props || props.length !== 2) {
                return window.dash_clientside.no_update;
            }
            
            // define properties of new points
            const size = 10
            const min_size = 1
            const max_size = size * 2 - 1
            const ys = [y, ...props[0]];
            const ws = [size, ...props[1].map((w) => Math.abs(w)*max_size + min_size)];
            const xs = Array(ys.length + 1).fill(x);
            const colors = ['red', ...props[1].map((w) => `hsl(${w < 0 ? 240 : 0}, ${Math.abs(w)*100}%, 50%)`)]
            
            // check last trace if it contains already some hovering points
            const n = figure.data.length;
            const trace = figure.data[n-1];
            
            if (!trace.name || trace.name !== 'hover-spatial-connections') {
                // no trace found; creating a new one
                Plotly.addTraces(figure, [{
                    x: xs, 
                    y: ys,
                    type: 'scatter',
                    name: 'hover-spatial-connections',
                    hoverinfo: 'skip',
                    marker: {
                        size: ws,
                        color: colors,
                        line: {'width': 0},
                        symbol: 'square',
                        opacity: 1.0
                    },
                    line: {
                        width: 1.0,
                        color: 'orange'
                    }
                }]);        
            } else {
                // found a trace; updating simply the points and colors
                Plotly.restyle(figure, {
                        x: [xs],
                        y: [ys],
                        'marker.color': [colors],
                        'marker.size': [ws],
                    }, n-1);        
            }
            
            // no update are needed in any case
            return window.dash_clientside.no_update;
        },

        subject_clear_out: function(id) {
            setTimeout(() => {
                graph = document.querySelector(`#${id} > .js-plotly-plot`);

                graph.on('plotly_unhover', (eventData) => {
                    if (!eventData || !eventData.event || eventData.event.type !== 'mouseout') {
                        return window.dash_clientside.no_update;
                    }

                    // check for a graph figure with specified id
                    const graphDiv = document.getElementById('st-graph');
                    const figure = graphDiv.querySelector('.js-plotly-plot');

                    if (!figure || !figure.data) {
                        return window.dash_clientside.no_update;
                    }

                    // check last trace if it contains already some hovering points
                    const n = figure.data.length;
                    const trace = figure.data[n-1];

                    if (trace.name || trace.name === 'hover-spatial-connections') {
                        Plotly.deleteTraces(figure, n-1);
                    }
                })
            }, 300);

            return window.dash_clientside.no_update;
        }
    }
});

