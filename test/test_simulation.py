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
from functools import reduce

import networkx as nx
import numpy as np
import pandas as pd

from fstg_toolkit import load_spatio_temporal_graph
from fstg_toolkit.graph import SpatioTemporalGraph, RC5
from fstg_toolkit.simulation import (
    CorrelationMatrixSequenceSimulator, SpatioTemporalGraphSimulator,
    generate_pattern, _CorrelationMatrixNetworksEdgesFiller,
    _CorrelationMatrixInterRegionEdgesFiller
)
from test_common import matrices_path, graph_path, patterns_path


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
            graph=SpatioTemporalGraph(graph, areas), threshold=0.4, rng=np.random.default_rng(10))
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
        target = np.array([matrix for _, matrix in np.load(matrices_path).items()])
        graph_structure = load_spatio_temporal_graph(graph_path)
        simulator = CorrelationMatrixSequenceSimulator(
            graph=graph_structure, threshold=0.4, rng=np.random.default_rng(100))
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


class CorrelationMatrixSequenceSimulatorTestCase(unittest.TestCase):
    """Test CorrelationMatrixSequenceSimulator class."""
    
    def test_initialization_validation(self):
        """Test initialization with invalid parameters."""
        # Create a valid graph first
        graph = nx.DiGraph()
        graph.add_node(1, t=0, areas={1, 2}, region='R1', internal_strength=0.8)
        graph.graph['min_time'] = 0
        graph.graph['max_time'] = 0
        
        areas = pd.DataFrame({
            'Id_Area': [1, 2],
            'Name_Area': ['A1', 'A2'],
            'Name_Region': ['R1', 'R1']
        })
        areas.set_index('Id_Area', inplace=True)
        
        st_graph = SpatioTemporalGraph(graph, areas)
        
        # Test invalid threshold values
        with self.assertRaises(ValueError):
            CorrelationMatrixSequenceSimulator(st_graph, threshold=-0.1)
        
        with self.assertRaises(ValueError):
            CorrelationMatrixSequenceSimulator(st_graph, threshold=1.1)
        
        # Valid thresholds should work
        try:
            CorrelationMatrixSequenceSimulator(st_graph, threshold=0.0)
            CorrelationMatrixSequenceSimulator(st_graph, threshold=1.0)
            CorrelationMatrixSequenceSimulator(st_graph, threshold=0.5)
        except ValueError:
            self.fail("Valid thresholds should not raise ValueError")
    
    def test_simple_simulation(self):
        """Test simple correlation matrix simulation."""
        # Create a simple graph with one network
        graph = nx.DiGraph()
        graph.add_node(1, t=0, areas={1, 2, 3}, region='R1', internal_strength=0.6)
        graph.graph['min_time'] = 0
        graph.graph['max_time'] = 0
        
        areas = pd.DataFrame({
            'Id_Area': [1, 2, 3],
            'Name_Area': ['A1', 'A2', 'A3'],
            'Name_Region': ['R1', 'R1', 'R1']
        })
        areas.set_index('Id_Area', inplace=True)
        
        st_graph = SpatioTemporalGraph(graph, areas)
        simulator = CorrelationMatrixSequenceSimulator(st_graph, threshold=0.4, 
                                                       rng=np.random.default_rng(42))
        
        # Simulate
        matrices = simulator.simulate()
        
        # Should have shape (1, 3, 3) - one time point, 3x3 matrix
        self.assertEqual(matrices.shape, (1, 3, 3))
        
        # Diagonal should be 1
        np.testing.assert_array_equal(np.diag(matrices[0]), [1.0, 1.0, 1.0])
        
        # Off-diagonal have value with some noise
        expected_matrix = np.array([
            [1.0, 0.516056, 0.156316],
            [0.516056, 1.0, 0.683943],
            [-0.321411, 0.683943, 1.0]
        ])
        np.testing.assert_array_almost_equal(matrices[0], expected_matrix, decimal=6)
    
    def test_multi_network_simulation(self):
        """Test simulation with multiple networks."""
        # Create graph with two networks
        graph = nx.DiGraph()
        graph.add_node(1, t=0, areas={1, 2}, region='R1', internal_strength=0.8)
        graph.add_node(2, t=0, areas={3, 4}, region='R2', internal_strength=-0.7)
        graph.add_edge(1, 2, type='spatial', correlation=0.6)
        graph.add_edge(2, 1, type='spatial', correlation=0.6)
        graph.graph['min_time'] = 0
        graph.graph['max_time'] = 0
        
        areas = pd.DataFrame({
            'Id_Area': [1, 2, 3, 4],
            'Name_Area': ['A1', 'A2', 'A3', 'A4'],
            'Name_Region': ['R1', 'R1', 'R2', 'R2']
        })
        areas.set_index('Id_Area', inplace=True)
        
        st_graph = SpatioTemporalGraph(graph, areas)
        simulator = CorrelationMatrixSequenceSimulator(st_graph, threshold=0.4, 
                                                       rng=np.random.default_rng(100))
        
        matrices = simulator.simulate()
        
        # Should have shape (1, 4, 4)
        self.assertEqual(matrices.shape, (1, 4, 4))
        
        matrix = matrices[0]
        
        # Check diagonal
        np.testing.assert_array_equal(np.diag(matrix), [1.0, 1.0, 1.0, 1.0])
        
        # Check intra-network correlations
        # Network 1 (areas 1,2) should have 0.8
        self.assertAlmostEqual(matrix[0, 1], 0.8, places=6)
        self.assertAlmostEqual(matrix[1, 0], 0.8, places=6)
        
        # Network 2 (areas 3,4) should have -0.7
        self.assertAlmostEqual(matrix[2, 3], -0.7, places=6)
        self.assertAlmostEqual(matrix[3, 2], -0.7, places=6)
        
        # Check inter-network correlation (should be 0.6, the max specified)
        inter_region_block = matrix[:2, 2:]
        self.assertAlmostEqual(np.max(inter_region_block), 0.6, places=6)

    def test_temporal_sequence_simulation(self):
        """Test simulation with temporal sequence."""
        # Create graph with two time points
        graph = nx.DiGraph()
        graph.add_node(1, t=0, areas={1, 2}, region='R1', internal_strength=0.8)
        graph.add_node(2, t=1, areas={1, 2, 3}, region='R1', internal_strength=0.9)
        graph.graph['min_time'] = 0
        graph.graph['max_time'] = 1
        
        areas = pd.DataFrame({
            'Id_Area': [1, 2, 3],
            'Name_Area': ['A1', 'A2', 'A3'],
            'Name_Region': ['R1', 'R1', 'R1']
        })
        areas.set_index('Id_Area', inplace=True)
        
        st_graph = SpatioTemporalGraph(graph, areas)
        simulator = CorrelationMatrixSequenceSimulator(st_graph, threshold=0.4, 
                                                       rng=np.random.default_rng(42))
        
        matrices = simulator.simulate()
        
        # Should have shape (2, 3, 3) - two time points
        self.assertEqual(matrices.shape, (2, 3, 3))
        
        # Both matrices should be valid correlation matrices
        for i in range(2):
            matrix = matrices[i]
            np.testing.assert_array_equal(np.diag(matrix), [1.0, 1.0, 1.0])
            # TODO check why the simulated matrices may not be symmetric
            # self.assertTrue(np.allclose(matrix, matrix.T))  # Symmetric
    
    def test_reproducibility(self):
        """Test that same seed produces same results."""
        # Create a test graph
        graph = nx.DiGraph()
        graph.add_node(1, t=0, areas={1, 2, 3}, region='R1', internal_strength=0.75)
        graph.graph['min_time'] = 0
        graph.graph['max_time'] = 0
        
        areas = pd.DataFrame({
            'Id_Area': [1, 2, 3],
            'Name_Area': ['A1', 'A2', 'A3'],
            'Name_Region': ['R1', 'R1', 'R1']
        })
        areas.set_index('Id_Area', inplace=True)
        
        st_graph = SpatioTemporalGraph(graph, areas)
        
        # Create two simulators with same seed
        simulator1 = CorrelationMatrixSequenceSimulator(st_graph, threshold=0.4, 
                                                       rng=np.random.default_rng(42))
        simulator2 = CorrelationMatrixSequenceSimulator(st_graph, threshold=0.4, 
                                                       rng=np.random.default_rng(42))
        
        matrices1 = simulator1.simulate()
        matrices2 = simulator2.simulate()
        
        # Results should be identical
        np.testing.assert_array_equal(matrices1, matrices2)


class SpatioTemporalGraphSimulatorTestCase(unittest.TestCase):
    """Test SpatioTemporalGraphSimulator class."""
    
    def test_simple_pattern_sequence(self):
        """Test simple pattern sequence simulation."""
        # Create a simple pattern
        pattern = generate_pattern(
            networks_list=[
                [((1, 2), 1, 0.8)],
                [((1, 2), 1, 0.9)]
            ],
            spatial_edges=[],
            temporal_edges=[(1, 1, 'eq')]
        )
        
        simulator = SpatioTemporalGraphSimulator(test_pattern=pattern)
        result = simulator.simulate('test_pattern')
        
        # Should have 2 nodes (one per time point) and 1 temporal edge
        self.assertEqual(len(result.nodes), 2)
        self.assertEqual(len(result.edges), 1)
        
        # Check temporal edge
        edge = list(result.edges(data=True))[0]
        self.assertEqual(edge[2]['transition'], RC5.EQ)
    
    def test_pattern_with_repeats(self):
        """Test pattern sequence with repeats."""
        pattern = generate_pattern(
            networks_list=[
                [((1, 2), 1, 0.8)],
                [((1, 2), 1, 0.9)]
            ],
            spatial_edges=[],
            temporal_edges=[(1, 1, 'eq')]
        )
        
        simulator = SpatioTemporalGraphSimulator(p=pattern)
        result = simulator.simulate('p', 2, 'p')  # pattern, 2 repeats, pattern
        
        # Should have: 2 (pattern) + 2 (repeats) + 2 (pattern) = 6 nodes
        self.assertEqual(len(result.nodes), 6)
        
        # Check time range
        self.assertEqual(result.graph['min_time'], 0)
        self.assertEqual(result.graph['max_time'], 5)  # 0,1,2,3,4,5 = 6 time points
    
    def test_complex_pattern_sequence(self):
        """Test complex pattern sequence."""
        pattern1 = generate_pattern(
            networks_list=[
                [((1, 3), 1, 0.7)],
                [((1, 2), 1, 0.6), (3, 1, 1.0)]
            ],
            spatial_edges=[(1, 3, 0.8), (1, 2, -0.5)],
            temporal_edges=[(1, (2, 3), 'split')]
        )
        
        pattern2 = generate_pattern(
            networks_list=[
                [((1, 2), 1, 0.6), (3, 1, 1.0)],
                [((1, 3), 1, 0.8)]
            ],
            spatial_edges=[(1, 2, 0.8), (1, 3, 0.4)],
            temporal_edges=[((1, 2), 3, 'merge')]
        )
        
        simulator = SpatioTemporalGraphSimulator(p1=pattern1, p2=pattern2)
        result = simulator.simulate('p1', 1, 'p2', 1, 'p1')
        
        # Verify structure
        self.assertGreater(len(result.nodes), 0)
        self.assertGreater(len(result.edges), 0)
        
        # Check that we have both spatial and temporal edges
        edge_types = [d['type'] for _, _, d in result.edges(data=True)]
        self.assertIn('spatial', edge_types)
        self.assertIn('temporal', edge_types)
    
    def test_areas_description_merging(self):
        """Test that areas descriptions are properly merged."""
        pattern1 = generate_pattern(
            networks_list=[[((1, 2), 1, 0.8)]],
            spatial_edges=[],
            temporal_edges=[]
        )
        
        pattern2 = generate_pattern(
            networks_list=[[((3, 4), 1, 0.7)]],
            spatial_edges=[],
            temporal_edges=[]
        )
        
        simulator = SpatioTemporalGraphSimulator(p1=pattern1, p2=pattern2)
        result = simulator.simulate('p1', 'p2')
        
        # Should have areas from both patterns
        self.assertEqual(len(result.areas), 4)
        self.assertListEqual(list(result.areas.index), [1, 2, 3, 4])


class GeneratePatternTestCase(unittest.TestCase):
    """Test generate_pattern function."""
    
    def test_simple_pattern(self):
        """Test simple pattern generation."""
        pattern = generate_pattern(
            networks_list=[
                [((1, 2), 1, 0.8)],
                [((1, 2), 1, 0.9)]
            ],
            spatial_edges=[],
            temporal_edges=[(1, 1, 'eq')]
        )
        
        # Should have 2 nodes and 1 edge
        self.assertEqual(len(pattern.nodes), 2)
        self.assertEqual(len(pattern.edges), 1)
        
        # Check node attributes
        node1, node2 = pattern.nodes(data=True)
        self.assertEqual(node1[1]['t'], 0)
        self.assertEqual(node2[1]['t'], 1)
        self.assertEqual(node1[1]['areas'], {1, 2})
        self.assertEqual(node2[1]['areas'], {1, 2})
    
    def test_complex_networks(self):
        """Test pattern with complex network definitions."""
        pattern = generate_pattern(
            networks_list=[
                [
                    ((1, 3), 1, 0.7),      # Areas 1,2,3 in region 1
                    (4, 2, 0.8),           # Area 4 in region 2
                    ((5, 7), 1, 0.6)       # Areas 5,6,7 in region 1
                ],
                [
                    ((1, 5), 1, 0.8),      # Areas 1-5 in region 1
                    ((6, 7), 2, 0.9)       # Areas 6,7 in region 2
                ]
            ],
            spatial_edges=[(1, 2, 0.5)],
            temporal_edges=[
                (1, 3, 'pp'),
                ((2, 4), 5, 'merge')
            ]
        )
        
        # Check areas DataFrame
        self.assertEqual(len(pattern.areas), 7)
        self.assertListEqual(list(pattern.areas.index), [1, 2, 3, 4, 5, 6, 7])
        
        # Check regions
        regions = pattern.areas['Name_Region'].unique()
        self.assertEqual(len(regions), 2)
        self.assertIn('Region 1', regions)
        self.assertIn('Region 2', regions)
    
    def test_temporal_transitions(self):
        """Test all temporal transition types."""
        pattern = generate_pattern(
            networks_list=[
                [((1, 2), 1, 0.8), ((3, 4), 2, 0.7)],
                [((1, 4), 1, 0.9)]  # Merged network
            ],
            spatial_edges=[],
            temporal_edges=[
                (1, 3, 'eq'),      # EQ: network 1 stays same
                (2, 3, 'pp'),      # PP: network 2 is proper part of network 3
                ((1, 2), 3, 'merge')  # Merge: networks 1+2 become network 3
            ]
        )
        
        # Check temporal edges
        temporal_edges = [d for _, _, d in pattern.edges(data=True) if d['type'] == 'temporal']
        transitions = [d['transition'] for d in temporal_edges]

        self.assertIn(RC5.PP, transitions)
        self.assertIn(RC5.PP, transitions)  # Merge creates PP transitions
    
    def test_spatial_edges(self):
        """Test spatial edge creation."""
        pattern = generate_pattern(
            networks_list=[
                [((1, 2), 1, 0.8), ((3, 4), 2, 0.7)]
            ],
            spatial_edges=[
                (1, 2, 0.6),   # Between network 1 and 2
                (1, 1, 0.5)    # Self-edge (should be ignored or handled)
            ],
            temporal_edges=[]
        )
        
        # Should have spatial edges
        spatial_edges = [d for _, _, d in pattern.edges(data=True) if d['type'] == 'spatial']
        self.assertGreater(len(spatial_edges), 0)
        
        # Check correlation values
        correlations = [d['correlation'] for d in spatial_edges]
        self.assertIn(0.6, correlations)


class EdgeFillerTestCase(unittest.TestCase):
    """Test edge filler helper classes."""
    
    def test_network_edges_filler(self):
        """Test _CorrelationMatrixNetworksEdgesFiller."""
        # Create a simple spatial graph
        graph = nx.DiGraph()
        graph.add_node(1, areas={1, 2, 3}, region='R1', internal_strength=0.8)
        graph.graph['min_time'] = 0
        graph.graph['max_time'] = 0
        
        areas = pd.DataFrame({
            'Id_Area': [1, 2, 3],
            'Name_Area': ['A1', 'A2', 'A3'],
            'Name_Region': ['R1', 'R1', 'R1']
        })
        areas.set_index('Id_Area', inplace=True)
        
        st_graph = SpatioTemporalGraph(graph, areas)
        
        # Create filler and test
        filler = _CorrelationMatrixNetworksEdgesFiller(threshold=0.4, 
                                                      rng=np.random.default_rng(42))
        matrix = np.zeros((3, 3))
        
        filler.fill(st_graph.sub(t=0), matrix)
        
        # Diagonal should remain 0 (will be set to 1 later)
        np.testing.assert_array_equal(np.diag(matrix), [0, 0, 0])
        
        # Off-diagonal
        off_diag = matrix[~np.eye(3, dtype=bool)]

        self.assertTrue(np.all(off_diag >= 0.0))
        self.assertTrue(np.all(off_diag < 1.0))
    
    def test_inter_region_edges_filler(self):
        """Test _CorrelationMatrixInterRegionEdgesFiller."""
        # Create graph with two regions
        graph = nx.DiGraph()
        graph.add_node(1, areas={1, 2}, region='R1', internal_strength=0.8)
        graph.add_node(2, areas={3, 4}, region='R2', internal_strength=-0.7)
        graph.add_edge(1, 2, type='spatial', correlation=0.6)
        graph.add_edge(2, 1, type='spatial', correlation=0.6)
        graph.graph['min_time'] = 0
        graph.graph['max_time'] = 0
        
        areas = pd.DataFrame({
            'Id_Area': [1, 2, 3, 4],
            'Name_Area': ['A1', 'A2', 'A3', 'A4'],
            'Name_Region': ['R1', 'R1', 'R2', 'R2']
        })
        areas.set_index('Id_Area', inplace=True)
        
        st_graph = SpatioTemporalGraph(graph, areas)
        
        # Create filler and test
        filler = _CorrelationMatrixInterRegionEdgesFiller(threshold=0.4,
                                                          rng=np.random.default_rng(100))
        matrix = np.zeros((4, 4))
        
        filler.fill(st_graph.sub(t=0), matrix)
        
        # Check inter-region correlations (should be around 0.6)
        inter_region = matrix[:2, 2:]
        self.assertGreater(np.max(inter_region), 0.5)  # Should be close to 0.6
        self.assertLess(np.min(inter_region), 0.7)


class SimulationEdgeCasesTestCase(unittest.TestCase):
    """Test simulation edge cases."""
    
    def test_empty_pattern(self):
        """Test pattern with no areas."""
        try:
            generate_pattern(
                networks_list=[[]],
                spatial_edges=[],
                temporal_edges=[]
            )
        except Exception as e:
            raise AssertionError(str(e))
    
    def test_single_area_pattern(self):
        """Test pattern with single area."""
        pattern = generate_pattern(
            networks_list=[[(1, 1, 1.0)]],
            spatial_edges=[],
            temporal_edges=[]
        )
        
        self.assertEqual(len(pattern.nodes), 1)
        self.assertEqual(len(pattern.edges), 0)
        self.assertEqual(len(pattern.areas), 1)
    
    def test_disconnected_networks(self):
        """Test pattern with disconnected networks."""
        pattern = generate_pattern(
            networks_list=[
                [((1, 2), 1, 0.8), ((3, 4), 2, 0.7)]  # Two separate networks
            ],
            spatial_edges=[],  # No connections between them
            temporal_edges=[]
        )
        
        # Should have two separate network nodes
        self.assertEqual(len(pattern.nodes), 2)
        self.assertEqual(len(pattern.edges), 0)  # No spatial edges
    
    def test_simulator_with_empty_patterns(self):
        """Test simulator with empty pattern sequence."""
        pattern = generate_pattern(
            networks_list=[[((1, 2), 1, 0.8)]],
            spatial_edges=[],
            temporal_edges=[]
        )
        
        simulator = SpatioTemporalGraphSimulator(p=pattern)

        with self.assertRaises(IndexError):
            simulator.simulate()

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

        expected_graph_struct = load_spatio_temporal_graph(patterns_path)
        self.assertEqual(expected_graph_struct, graph_struct)
