import unittest

from fmri_st_graph import load_spatio_temporal_graph, CorrelationMatrixSequenceSimulator, \
    spatio_temporal_graph_from_corr_matrices
from fmri_st_graph.graph import are_st_graphs_close


class SpatioTemporalGraphFactoryTestCase(unittest.TestCase):
    def test_spatio_temporal_graph_from_corr_matrices(self):
        expected_graph = load_spatio_temporal_graph('data/toy-example_graph.zip')
        corr_matrices = CorrelationMatrixSequenceSimulator(expected_graph).simulate()
        graph = spatio_temporal_graph_from_corr_matrices(corr_matrices, expected_graph.areas)
        self.assertTrue(are_st_graphs_close(expected_graph, graph))
