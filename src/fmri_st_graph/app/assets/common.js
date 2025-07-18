window.dash_clientside = Object.assign({}, window.dash_clientside, {
    clientside: {
        subject_node_hover: function(hoverData, storeHoverGraph) {
            if (!hoverData || !hoverData.points || hoverData.points.length === 0 || !storeHoverGraph) {
                return window.dash_clientside.no_update;
            }
        
            const graphDiv = document.getElementById('st-graph');
            const figure = graphDiv.querySelector('.js-plotly-plot');
        
            if (!figure || !figure.data) {
                return window.dash_clientside.no_update;
            }
            
            const point = hoverData.points[0];
            const x = point['x'];
            const y = point['y'];
            
            const coord = storeHoverGraph[x][y];
            // TODO check that coord is has two components
            const xs = [x, ...coord[0]];
            const ys = [y, ...coord[1]];
            
            const colors = Array(xs.length + 1).fill('green');
            colors[0] = 'red'; // Highlight the hovered point
            
            const n = figure.data.length;
            const trace = figure.data[n-1];
            
            if (!trace.name || trace.name !== 'hover-spatial-connections') {
                Plotly.addTraces(figure, [{
                    x: xs, 
                    y: ys,
                    type: 'scatter',
                    name: 'hover-spatial-connections',
                    hoverinfo: 'skip',
                    marker: {
                        size: 12,
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
                Plotly.restyle(figure, {
                        x: [xs],
                        y: [ys],
                        'marker.color': [colors],
                    }, n-1);        
            }
            
            return window.dash_clientside.no_update;
        }
    }
});

