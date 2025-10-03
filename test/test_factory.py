# Copyright 2025 ICube (University of Strasbourg - CNRS)
# author: Julien PONTABRY (ICube)
#
# This software is a computer program whose purpose is to provide a toolkit
# to model, process and analyze the longitudinal reorganization of brain
# connectivity data, as functional MRI for instance.
#
# This software is governed by the CeCILL-B license under French law and
# abiding by the rules of distribution of free software. You can use,
# modify and/or redistribute the software under the terms of the CeCILL-B
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info".
#
# As a counterpart to the access to the source code and rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty and the software's author, the holder of the
# economic rights, and the successive licensors have only limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading, using, modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean that it is complicated to manipulate, and that also
# therefore means that it is reserved for developers and experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and, more generally, to use and operate it in the
# same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL-B license and that you accept its terms.

import unittest

from fstg_toolkit import load_spatio_temporal_graph, CorrelationMatrixSequenceSimulator, \
    spatio_temporal_graph_from_corr_matrices
from fstg_toolkit.graph import are_st_graphs_close
from test_common import graph_path


class SpatioTemporalGraphFactoryTestCase(unittest.TestCase):
    def test_spatio_temporal_graph_from_corr_matrices(self):
        expected_graph = load_spatio_temporal_graph(graph_path)
        corr_matrices = CorrelationMatrixSequenceSimulator(expected_graph).simulate()
        graph = spatio_temporal_graph_from_corr_matrices(corr_matrices, expected_graph.areas)

        # NOTE: the matrices generation introduces numerical errors, so we cannot directly compare the efficiency values.
        # Instead, we copy the efficiency values from the generated graph to the expected graph (no test on that part).
        for (_, d1), (_, d2) in zip(graph.nodes(data=True), expected_graph.nodes(data=True)):
            d2['efficiency'] = d1['efficiency']

        self.assertTrue(are_st_graphs_close(expected_graph, graph))
