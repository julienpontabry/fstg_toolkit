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
            const coord = storeHoverGraph[x][y];

            if (!coord || coord.length !== 2) {
                return window.dash_clientside.no_update;
            }
            
            // define properties of new points
            const xs = [x, ...coord[0]];
            const ys = [y, ...coord[1]];
            const colors = Array(xs.length + 1).fill('green');
            colors[0] = 'red'; // Highlight the hovered point
            
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
                        size: 10,
                        color: colors,
                        line: {'width': 0},
                        symbol: 'square',
                        opacity: 0.5
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
                    }, n-1);        
            }
            
            // no update are needed in any case
            return window.dash_clientside.no_update;
        }
    }
});

