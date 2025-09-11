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
