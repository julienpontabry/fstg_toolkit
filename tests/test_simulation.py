import unittest

import networkx as nx
import numpy as np
import pandas as pd

from fmri_st_graph import SpatioTemporalGraph, load_spatio_temporal_graph
from fmri_st_graph.simulation import CorrelationMatrixSequenceSimulator

class CorrelationMatrixSimulationTestCase(unittest.TestCase):
    @staticmethod
    def test_corr_matrix_sequence_simulation():
        target = np.array([matrix for _, matrix in np.load('data/toy-example_matrices.npz').items()])
        graph_structure = load_spatio_temporal_graph('data/toy-example_graph.zip')
        simulator = CorrelationMatrixSequenceSimulator(
            graph_struct=graph_structure, threshold=0.4, rng=np.random.default_rng(100))
        matrices = simulator.simulate()
        np.testing.assert_array_equal(target, matrices)

    def test_corr_matrix_single_simulation(self):
        n1_is = 0.98
        n2_is = -0.98
        n_corr = 0.94

        graph = nx.DiGraph()
        graph.add_node(1, t=0, areas={1, 2}, region='Region 1', internal_strength=n1_is)
        graph.add_node(2, t=0, areas={3, 4}, region='Region 2', internal_strength=n2_is)
        graph.add_edge(1, 2, correlation=n_corr, t=0, type='spatial')
        graph.add_edge(2, 1, correlation=n_corr, t=0, type='spatial')
        graph.graph['min_time'] = 0
        graph.graph['max_time'] = 0

        areas = pd.DataFrame({'Id_Area': [1, 2, 3, 4],
                              'Name_Area': ['A1', 'A2', 'A3', 'A4'],
                              'Name_Region': ['R1', 'R1', 'R2', 'R2']})
        areas.set_index('Id_Area', inplace=True)

        simulator = CorrelationMatrixSequenceSimulator(
            graph_struct=SpatioTemporalGraph(graph, areas), threshold=0.4, rng=np.random.default_rng(10))
        matrix = simulator.simulate()[0]

        self.assertEqual(2, len(matrix.shape))
        for i in range(2):
            self.assertEqual(4, matrix.shape[i])

        self.assertEqual(n1_is, matrix[0, 1])
        self.assertEqual(n1_is, matrix[1, 0])

        self.assertEqual(n2_is, matrix[2, 3])
        self.assertEqual(n2_is, matrix[3, 2])

        self.assertEqual(n_corr, matrix[2:, :2].max())
        self.assertEqual(n_corr, matrix[:2, 2:].max())

        for i in range(4):
            self.assertEqual(1, matrix[i, i])
