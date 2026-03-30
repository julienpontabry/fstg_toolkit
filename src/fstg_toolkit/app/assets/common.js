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

let _hoverDebounceTimer = null;

window.dash_clientside = Object.assign({}, window.dash_clientside, {
    clientside: {
        subject_node_hover: function(hoverData, storeHoverGraph) {
            if (!hoverData || !hoverData.points || hoverData.points.length === 0 || !storeHoverGraph) {
                return window.dash_clientside.no_update;
            }

            clearTimeout(_hoverDebounceTimer);
            _hoverDebounceTimer = setTimeout(() => {
                const graphDiv = document.getElementById('st-graph');
                const figure = graphDiv && graphDiv.querySelector('.js-plotly-plot');
                if (!figure || !figure.data) return;

                const point = hoverData.points[0];
                const props = storeHoverGraph[point.x] && storeHoverGraph[point.x][point.y];
                if (!props || props.length !== 2) return;

                const size = 10, min_size = 1, max_size = size * 2 - 1;
                const ys = [point.y, ...props[0]];
                const ws = [size, ...props[1].map(w => Math.abs(w) * max_size + min_size)];
                const xs = Array(ys.length).fill(point.x);
                const colors = ['black', ...props[1].map(w =>
                    `hsl(${w < 0 ? 240 : 0}, ${Math.abs(w) * 100}%, 50%)`
                )];

                Plotly.restyle(figure, {
                    x: [xs], y: [ys],
                    'marker.color': [colors],
                    'marker.size': [ws],
                }, figure.data.length - 1);
            }, 16);  // ~60 fps cap

            return window.dash_clientside.no_update;
        },

        subject_clear_out: function(id) {
            setTimeout(() => {
                const graph = document.querySelector(`#${id} > .js-plotly-plot`);

                graph.on('plotly_unhover', (eventData) => {
                    if (!eventData || !eventData.event || eventData.event.type !== 'mouseout') return;

                    const graphDiv = document.getElementById('st-graph');
                    const figure = graphDiv && graphDiv.querySelector('.js-plotly-plot');
                    if (!figure || !figure.data) return;

                    const n = figure.data.length;
                    if (figure.data[n-1].name === 'hover-spatial-connections') {
                        Plotly.restyle(figure, { x: [[]], y: [[]], 'marker.size': [[]], 'marker.color': [[]] }, n-1);
                    }
                });
            }, 300);

            return window.dash_clientside.no_update;
        }
    }
});

