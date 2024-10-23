import unittest
from functools import reduce

import networkx as nx
import numpy as np
import pandas as pd

from fmri_st_graph import SpatioTemporalGraph, load_spatio_temporal_graph, generate_pattern, \
    SpatioTemporalGraphSimulator, save_spatio_temporal_graph
from fmri_st_graph.graph import RC5
from fmri_st_graph.simulation import CorrelationMatrixSequenceSimulator


class CorrelationMatrixSimulationTestCase(unittest.TestCase):
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

    @staticmethod
    def test_corr_matrix_sequence_simulation():
        target = np.array([matrix for _, matrix in np.load('data/toy-example_matrices.npz').items()])
        graph_structure = load_spatio_temporal_graph('data/toy-example_graph.zip')
        simulator = CorrelationMatrixSequenceSimulator(
            graph_struct=graph_structure, threshold=0.4, rng=np.random.default_rng(100))
        matrices = simulator.simulate()
        np.testing.assert_array_equal(target, matrices)


class SpatioTemporalGraphSimulationTestCase(unittest.TestCase):
    def test_generate_pattern(self):
        pattern = generate_pattern(
            networks_list=[[((1, 5), 1, -0.2), ((6, 7), 2, 0.3), ((8, 10), 2, 0.6)],
                         [((1, 5), 1, 0.6), ((6, 10), 2, -0.5)]],
            spatial_edges=[(1, 2, 0.45), (4, 5, 0.8)],
            temporal_edges=[(1, 4, 'eq'), ((2, 3), 5, 'merge')])

        id_list = list(range(1, 11))
        expected_areas = pd.DataFrame({'Id_Area': id_list,
                                 'Name_Area': [f"Area {i}" for i in id_list],
                                 'Name_Region': ["Region 1"]*5 + ["Region 2"]*5})
        expected_areas.set_index('Id_Area', inplace=True)

        expected_graph = nx.Graph()
        expected_graph.add_nodes_from([
            (1, dict(t=0, areas={1, 2, 3, 4, 5}, region="Region 1", internal_strength=-0.2)),
            (2, dict(t=0, areas={6, 7}, region="Region 2", internal_strength=0.3)),
            (3, dict(t=0, areas={8, 9, 10}, region="Region 2", internal_strength=0.6)),
            (4, dict(t=1, areas={1, 2, 3, 4, 5}, region="Region 1", internal_strength=0.6)),
            (5, dict(t=1, areas={6, 7, 8, 9, 10}, region="Region 2", internal_strength=-0.5))])
        expected_graph.add_edges_from([
            (1, 2, dict(correlation=0.45, type='spatial')),
            (4, 5, dict(correlation=0.8, type='spatial'))])
        expected_graph = nx.DiGraph(expected_graph)
        expected_graph.add_edges_from([
            (1, 4, dict(transition=RC5.EQ, type='temporal')),
            (2, 5, dict(transition=RC5.PP, type='temporal')),
            (3, 5, dict(transition=RC5.PP, type='temporal'))])
        expected_graph.graph = dict(min_time=0, max_time=1)

        self.assertEqual(SpatioTemporalGraph(expected_graph, expected_areas), pattern)

    def test_simulation_from_simple_patterns(self):
        pattern1 = generate_pattern(
            networks_list=[[((1, 3), 1, 0.8), ((4, 5), 2, -0.8)]],
            spatial_edges=[(1, 2, 0.6)],
            temporal_edges=[])

        pattern2 = generate_pattern(
            networks_list=[
                [((1, 3), 1, 0.8), ((4, 5), 2, -0.8)],
                [((1, 2), 1, 0.7), (3, 1, 1), ((4, 5), 2, -0.8)]],
            spatial_edges=[(1, 2, 0.6), (3, 5, 0.5)],
            temporal_edges=[(1, (3, 4), 'split'), (2, 5, 'eq')])

        pattern3 = generate_pattern(
            networks_list=[
                [((1, 2), 1, 0.7), (3, 1, 1), ((4, 5), 2, -0.8)],
                [((1, 3), 1, 0.8), ((4, 5), 2, -0.8)]],
            spatial_edges=[(1, 3, 0.5), (4, 5, 0.6)],
            temporal_edges=[((1, 2), 4, 'merge'), (3, 5, 'eq')])

        simulator = SpatioTemporalGraphSimulator(p1=pattern1, p2=pattern2, p3=pattern3)
        graph_struct = simulator.simulate('p2', 10, 'p3', 5, 'p1')

        expected_graph = nx.Graph()
        expected_graph.graph = {'min_time': 0, 'max_time': 19}

        expected_graph.add_nodes_from([
            (1, dict(t=0, areas={1, 2, 3}, region="Region 1", internal_strength=0.8)),
            (2, dict(t=0, areas={4, 5}, region="Region 2", internal_strength=-0.8)),
            (3, dict(t=1, areas={1, 2}, region="Region 1", internal_strength=0.7)),
            (4, dict(t=1, areas={3}, region="Region 1", internal_strength=1)),
            (5, dict(t=1, areas={4, 5}, region="Region 2", internal_strength=-0.8))])
        expected_graph.add_nodes_from(reduce(list.__add__, [
            [(6+3*i, dict(t=2+i, areas={1, 2}, region="Region 1", internal_strength=0.7)),
             (7+3*i, dict(t=2+i, areas={3}, region="Region 1", internal_strength=1)),
             (8+3*i, dict(t=2+i, areas={4, 5}, region="Region 2", internal_strength=-0.8))]
            for i in range(10)], []))
        expected_graph.add_nodes_from([
            (36, dict(t=12, areas={1, 2}, region="Region 1", internal_strength=0.7)),
            (37, dict(t=12, areas={3}, region="Region 1", internal_strength=1)),
            (38, dict(t=12, areas={4, 5}, region="Region 2", internal_strength=-0.8)),
            (39, dict(t=13, areas={1, 2, 3}, region="Region 1", internal_strength=0.8)),
            (40, dict(t=13, areas={4, 5}, region="Region 2", internal_strength=-0.8))])
        expected_graph.add_nodes_from(reduce(list.__add__, [
            [(41+2*i, dict(t=14+i, areas={1, 2, 3}, region="Region 1", internal_strength=0.8)),
             (42+2*i, dict(t=14+i, areas={4, 5}, region="Region 2", internal_strength=-0.8))]
            for i in range(5)], []))
        expected_graph.add_nodes_from([
            (51, dict(t=19, areas={1, 2, 3}, region="Region 1", internal_strength=0.8)),
            (52, dict(t=19, areas={4, 5}, region="Region 2", internal_strength=-0.8))])

        expected_graph.add_edges_from([
            (1, 2, dict(correlation=0.6, type='spatial')),
            (3, 5, dict(correlation=0.5, type='spatial'))])
        expected_graph.add_edges_from([
            (6+3*i, 8+3*i, dict(correlation=0.5, type='spatial'))
            for i in range(10)])
        expected_graph.add_edges_from([
            (36, 38, dict(correlation=0.5, type='spatial')),
            (39, 40, dict(correlation=0.6, type='spatial'))])
        expected_graph.add_edges_from([
            (41+2*i, 42+2*i, dict(correlation=0.6, type='spatial'))
            for i in range(5)])
        expected_graph.add_edges_from([
            (51, 52, dict(correlation=0.6, type='spatial'))])

        expected_graph = nx.DiGraph(expected_graph)

        expected_graph.add_edges_from([
            (1, 3, dict(transition=RC5.PPi, type='temporal')),
            (1, 4, dict(transition=RC5.PPi, type='temporal')),
            (2, 5, dict(transition=RC5.EQ, type='temporal'))])
        expected_graph.add_edges_from(reduce(list.__add__, [
            [(3+3*i, 6+3*i, dict(transition=RC5.EQ, type='temporal')),
             (4+3*i, 7+3*i, dict(transition=RC5.EQ, type='temporal')),
             (5+3*i, 8+3*i, dict(transition=RC5.EQ, type='temporal'))]
            for i in range(11)], []))
        expected_graph.add_edges_from([
            (36, 39, dict(transition=RC5.PP, type='temporal')),
            (37, 39, dict(transition=RC5.PP, type='temporal')),
            (38, 40, dict(transition=RC5.EQ, type='temporal'))])
        expected_graph.add_edges_from(reduce(list.__add__, [
            [(39+2*i, 41+2*i, dict(transition=RC5.EQ, type='temporal')),
             (40+2*i, 42+2*i, dict(transition=RC5.EQ, type='temporal'))]
            for i in range(6)], []))

        expected_areas = pd.DataFrame({
            'Id_Area': list(range(1, 6)),
            'Name_Area': [f"Area {i+1}" for i in range(5)],
            'Name_Region': ["Region 1"]*3 + ["Region 2"]*2})
        expected_areas.set_index('Id_Area', inplace=True)

        self.assertEqual(SpatioTemporalGraph(expected_graph, expected_areas), graph_struct)

    def test_simulation_from_complex_patterns(self):
        pattern1 = generate_pattern(
            networks_list=[
                [((1, 5), 1, 0.8), ((6, 10), 2, -0.5)],
                [((1, 3), 1, 0.6), ((4, 5), 1, 0.7), ((6, 10), 2, 0.7)],
                [((1, 5), 1, -0.45), ((6, 7), 2, 0.42), ((8, 10), 2, 0.9)]],
            spatial_edges=[(1, 2, 0.5), (3, 5, 0.6), (6, 7, 0.6), (6, 8, -0.5)],
            temporal_edges=[(1, (3, 4), 'split'), (2, 5, 'eq'),
                            ((3, 4), 6, 'merge'), (5, (7, 8), 'split')])

        pattern2 = generate_pattern(
            networks_list=[[
                ((1, 5), 1, -0.2), ((6, 7), 2, 0.3), ((8, 10), 2, 0.6)],
                [((1, 5), 1, 0.6), ((6, 10), 2, -0.5)]],
            spatial_edges=[(1, 2, 0.45), (4, 5, 0.8)],
            temporal_edges=[(1, 4, 'eq'), ((2, 3), 5, 'merge')])

        simulator = SpatioTemporalGraphSimulator(p1=pattern1, p2=pattern2)
        graph_struct = simulator.simulate('p1', 5, 'p2', 'p1', 10, 'p2', 'p1', 2, 'p2')

        expected_graph_struct = load_spatio_temporal_graph('data/patterns-example.zip')
        self.assertEqual(expected_graph_struct, graph_struct)
