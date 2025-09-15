// Copyright 2025 ICube (University of Strasbourg - CNRS)
// author: Julien PONTABRY (ICube)
//
// This software is a computer program whose purpose is to provide a toolkit
// to model, process and analyze the longitudinal reorganization of brain
// connectivity data, as functional MRI for instance.
//
// This software is governed by the CeCILL-B license under French law and
// abiding by the rules of distribution of free software. You can use,
// modify and/or redistribute the software under the terms of the CeCILL-B
// license as circulated by CEA, CNRS and INRIA at the following URL
// "http://www.cecill.info".
//
// As a counterpart to the access to the source code and rights to copy,
// modify and redistribute granted by the license, users are provided only
// with a limited warranty and the software's author, the holder of the
// economic rights, and the successive licensors have only limited
// liability.
//
// In this respect, the user's attention is drawn to the risks associated
// with loading, using, modifying and/or developing or reproducing the
// software by the user in light of its specific status of free software,
// that may mean that it is complicated to manipulate, and that also
// therefore means that it is reserved for developers and experienced
// professionals having in-depth computer knowledge. Users are therefore
// encouraged to load and test the software's suitability as regards their
// requirements in conditions enabling the security of their systems and/or
// data to be ensured and, more generally, to use and operate it in the
// same conditions as regards security.
//
// The fact that you are presently reading this means that you have had
// knowledge of the CeCILL-B license and that you accept its terms.

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
            const colors = ['black', ...props[1].map((w) => `hsl(${w < 0 ? 240 : 0}, ${Math.abs(w)*100}%, 50%)`)]
            
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

