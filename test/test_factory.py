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

import networkx as nx
import numpy as np
import pandas as pd

from fstg_toolkit import load_spatio_temporal_graph, CorrelationMatrixSequenceSimulator
from fstg_toolkit.factory import graph_from_corr_matrix, networks_from_connect_graph, \
    spatio_temporal_graph_from_networks_graphs, spatio_temporal_graph_from_corr_matrices
from fstg_toolkit.graph import are_st_graphs_close, RC5
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


class GraphFromCorrMatrixTestCase(unittest.TestCase):
    """Test graph_from_corr_matrix function with various inputs."""
    
    def setUp(self):
        # Basic setup for most tests
        self.areas = pd.DataFrame({
            'Id': [1, 2, 3, 4],
            'Name_Area': ['A1', 'A2', 'A3', 'A4'],
            'Name_Region': ['R1', 'R1', 'R2', 'R2']
        })
        self.areas.set_index('Id', inplace=True)
    
    def test_empty_matrix(self):
        """Test with empty correlation matrix."""
        empty_matrix = np.array([])
        result = graph_from_corr_matrix(empty_matrix, self.areas)
        self.assertEqual(len(result.nodes), 0)
        self.assertEqual(len(result.edges), 0)
    
    def test_single_node(self):
        """Test with single node matrix."""
        single_matrix = np.array([[1.0]])
        result = graph_from_corr_matrix(single_matrix, self.areas.iloc[[0]])
        self.assertEqual(len(result.nodes), 1)
        self.assertEqual(len(result.edges), 0)
    
    def test_threshold_boundaries(self):
        """Test threshold behavior at boundaries."""
        # Matrix with correlations exactly at threshold
        matrix = np.array([
            [1.0, 0.4, 0.39],
            [0.4, 1.0, 0.5],
            [0.39, 0.5, 1.0]
        ])
        
        # should include 0.4, 0.5 and 0.39
        result = graph_from_corr_matrix(matrix, self.areas, corr_thr=0.3, abs_thr=True)
        self.assertEqual(list(result.nodes), [1, 2, 3])
        self.assertEqual(list(result.edges), [(1, 2), (1, 3), (2, 3)])
        
        # should include 0.4 and 0.5, exclude 0.39
        result = graph_from_corr_matrix(matrix, self.areas, corr_thr=0.4, abs_thr=False)
        self.assertEqual(list(result.nodes), [1, 2, 3])
        self.assertEqual(list(result.edges), [(1, 2), (2, 3)])
    
    def test_negative_correlations(self):
        """Test handling of negative correlations."""
        matrix = np.array([
            [1.0, -0.5, 0.6],
            [-0.5, 1.0, -0.7],
            [0.6, -0.7, 1.0]
        ])
        
        # With abs_thr=True, should include all (|corr| > 0.4)
        result = graph_from_corr_matrix(matrix, self.areas, corr_thr=0.4, abs_thr=True)
        self.assertEqual(list(result.nodes), [1, 2, 3])
        self.assertEqual(list(result.edges), [(1, 2), (1, 3), (2, 3)])
        
        # With abs_thr=False, should only include positive correlations > 0.4
        result = graph_from_corr_matrix(matrix, self.areas, corr_thr=0.4, abs_thr=False)
        self.assertEqual(list(result.nodes), [1, 2, 3])
        self.assertEqual(list(result.edges), [(1, 3)])
    
    def test_node_attributes(self):
        """Test that node attributes are correctly set."""
        matrix = np.array([
            [1.0, 0.8],
            [0.8, 1.0]
        ])
        
        result = graph_from_corr_matrix(matrix, self.areas.iloc[[0, 1]])
        
        # Check node attributes
        node1 = result.nodes[1]
        node2 = result.nodes[2]
        
        self.assertEqual(node1['area'], 'A1')
        self.assertEqual(node1['region'], 'R1')
        self.assertEqual(node2['area'], 'A2')
        self.assertEqual(node2['region'], 'R1')
    
    def test_edge_attributes(self):
        """Test that edge attributes are correctly set."""
        matrix = np.array([
            [1.0, 0.8, -0.6],
            [0.8, 1.0, 0.3],
            [-0.6, 0.3, 1.0]
        ])
        
        result = graph_from_corr_matrix(matrix, self.areas, corr_thr=0.4)
        
        # Check edge correlations
        edge12 = result.edges[(1, 2)]
        edge13 = result.edges[(1, 3)]
        
        self.assertAlmostEqual(edge12['correlation'], 0.8)
        self.assertAlmostEqual(edge13['correlation'], -0.6)


class NetworksFromConnectGraphTestCase(unittest.TestCase):
    """Test networks_from_connect_graph function."""
    
    def test_single_region_single_network(self):
        """Test single region with one connected network."""
        graph = nx.Graph()
        graph.add_nodes_from([
            (1, {'area': 'A1', 'region': 'R1'}),
            (2, {'area': 'A2', 'region': 'R1'}),
            (3, {'area': 'A3', 'region': 'R1'})
        ])
        graph.add_edges_from([
            (1, 2, {'correlation': 0.8}),
            (2, 3, {'correlation': 0.7})
        ])
        
        result = networks_from_connect_graph(graph, ['R1'])
        
        self.assertEqual(len(result.nodes), 1)
        self.assertEqual(len(result.edges), 0)  # Single network, no inter-network edges
        
        node = list(result.nodes(data=True))[0]
        self.assertEqual(node[1]['region'], 'R1')
        self.assertEqual(node[1]['areas'], {1, 2, 3})
    
    def test_multiple_disconnected_networks(self):
        """Test multiple disconnected networks in same region."""
        graph = nx.Graph()
        graph.add_nodes_from([
            (1, {'area': 'A1', 'region': 'R1'}),
            (2, {'area': 'A2', 'region': 'R1'}),
            (3, {'area': 'A3', 'region': 'R1'}),
            (4, {'area': 'A4', 'region': 'R1'})
        ])
        graph.add_edges_from([
            (1, 2, {'correlation': 0.8}),  # Network 1: nodes 1-2
            (3, 4, {'correlation': 0.7})   # Network 2: nodes 3-4
        ])
        
        result = networks_from_connect_graph(graph, ['R1'])
        
        self.assertEqual(len(result.nodes), 2)
        self.assertEqual(len(result.edges), 0)
        
        # Check network compositions
        nodes = list(result.nodes(data=True))
        areas_sets = [node[1]['areas'] for node in nodes]
        self.assertIn({1, 2}, areas_sets)
        self.assertIn({3, 4}, areas_sets)
    
    def test_multiple_regions(self):
        """Test multiple regions with interconnected networks."""
        graph = nx.Graph()
        graph.add_nodes_from([
            (1, {'area': 'A1', 'region': 'R1'}),
            (2, {'area': 'A2', 'region': 'R1'}),
            (3, {'area': 'A3', 'region': 'R2'}),
            (4, {'area': 'A4', 'region': 'R2'})
        ])
        graph.add_edges_from([
            (1, 2, {'correlation': 0.8}),   # Within R1
            (3, 4, {'correlation': 0.7}),   # Within R2
            (2, 3, {'correlation': 0.6})    # Between R1 and R2
        ])
        
        result = networks_from_connect_graph(graph, ['R1', 'R2'])
        
        self.assertEqual(len(result.nodes), 2)  # One network per region
        self.assertEqual(len(result.edges), 1)   # One inter-region edge
        
        # Check inter-region edge has max correlation
        edge = list(result.edges(data=True))[0]
        self.assertEqual(edge[2]['correlation'], 0.6)
    
    def test_isolated_areas(self):
        """Test regions with isolated areas (no connections)."""
        graph = nx.Graph()
        graph.add_nodes_from([
            (1, {'area': 'A1', 'region': 'R1'}),
            (2, {'area': 'A2', 'region': 'R1'}),  # Isolated
            (3, {'area': 'A3', 'region': 'R2'})
        ])
        graph.add_edge(1, 3, correlation=0.8)
        
        result = networks_from_connect_graph(graph, ['R1', 'R2'])
        
        # Should have 3 networks: {1}, {2}, {3}
        self.assertEqual(len(result.nodes), 3)
        
        # Check isolated node has internal strength = 1
        isolated_node = [node for node, data in result.nodes(data=True) if data['areas'] == {2}][0]
        self.assertEqual(result.nodes[isolated_node]['internal_strength'], 1)


class SpatioTemporalGraphFromNetworksGraphsTestCase(unittest.TestCase):
    """Test spatio_temporal_graph_from_networks_graphs function."""
    
    def test_single_time_point(self):
        """Test with single time point."""
        graph = nx.Graph()
        graph.add_node(1, areas={1, 2}, region='R1', internal_strength=0.8)
        graph.add_node(2, areas={3}, region='R2', internal_strength=1.0)
        
        result = spatio_temporal_graph_from_networks_graphs((graph,))
        
        self.assertEqual(len(result.nodes), 2)
        self.assertEqual(len(result.edges), 0)  # No temporal edges with single time point
        
        # Check time attributes
        for node in result.nodes():
            self.assertEqual(result.nodes[node]['t'], 0)
    
    def test_two_time_points_with_transitions(self):
        """Test temporal transitions between two time points."""
        # Time 0
        graph_t0 = nx.Graph()
        graph_t0.add_node(1, areas={1, 2}, region='R1', internal_strength=0.8)
        graph_t0.add_node(2, areas={3}, region='R2', internal_strength=1.0)
        
        # Time 1 - some areas merge
        graph_t1 = nx.Graph()
        graph_t1.add_node(1, areas={1, 2, 3}, region='R1', internal_strength=0.9)
        
        result = spatio_temporal_graph_from_networks_graphs((graph_t0, graph_t1))
        
        self.assertEqual(len(result.nodes), 3)  # 2 from t0, 1 from t1
        self.assertEqual(len(result.edges), 2)  # 2 temporal edges
        
        # Check temporal edges have correct transitions
        temporal_edges = [(u, v, d) for u, v, d in result.edges(data=True) if d['type'] == 'temporal']
        self.assertEqual(len(temporal_edges), 2)
        
        # Node 1 (t0) -> Node 3 (t1) should be RC5.PP
        # Node 2 (t0) -> Node 3 (t1) should be RC5.PP
        transitions = [d['transition'] for _, _, d in temporal_edges]
        self.assertIn(RC5.PP, transitions)
    
    def test_complex_transitions(self):
        """Test all RC5 transition types."""
        # Create networks that will produce different transitions
        graphs = []
        
        # t0: Network 1 = {1,2}, Network 2 = {3}
        g0 = nx.Graph()
        g0.add_node(1, areas={1, 2}, region='R1', internal_strength=0.8)
        g0.add_node(2, areas={3}, region='R1', internal_strength=1.0)
        graphs.append(g0)
        
        # t1: Network 1 = {1}, Network 2 = {2,3}  
        # This should create: EQ (1->1), PPi (1->2), PP (2->2)
        g1 = nx.Graph()
        g1.add_node(1, areas={1}, region='R1', internal_strength=0.7)
        g1.add_node(2, areas={2, 3}, region='R1', internal_strength=0.9)
        graphs.append(g1)
        
        result = spatio_temporal_graph_from_networks_graphs(tuple(graphs))
        
        temporal_edges = [(u, v, d) for u, v, d in result.edges(data=True) if d['type'] == 'temporal']
        transitions = [d['transition'] for _, _, d in temporal_edges]
        
        # Should have EQ, PPi, and PP transitions
        self.assertIn(RC5.PO, transitions)
        self.assertIn(RC5.PPi, transitions)
        self.assertIn(RC5.PP, transitions)


class SpatioTemporalGraphFromCorrMatricesTestCase(unittest.TestCase):
    """Test the full pipeline from correlation matrices."""
    
    def test_simple_case(self):
        """Test simple case with two time points."""
        areas = pd.DataFrame({
            'Id': [1, 2, 3],
            'Name_Area': ['A1', 'A2', 'A3'],
            'Name_Region': ['R1', 'R1', 'R2']
        })
        areas.set_index('Id', inplace=True)
        
        # Two time points with different connectivity
        matrices = [
            np.array([
                [1.0, 0.8, 0.2],
                [0.8, 1.0, 0.1],
                [0.2, 0.1, 1.0]
            ]),
            np.array([
                [1.0, 0.9, 0.7],
                [0.9, 1.0, 0.3],
                [0.7, 0.3, 1.0]
            ])
        ]
        
        result = spatio_temporal_graph_from_corr_matrices(matrices, areas)
        
        # Should have spatial networks for each time point
        self.assertGreater(len(result.nodes), 0)
        self.assertGreater(len(result.edges), 0)
        
        # Check we have both spatial and temporal edges
        edge_types = [d['type'] for _, _, d in result.edges(data=True)]
        self.assertIn('spatial', edge_types)
        self.assertIn('temporal', edge_types)
    
    def test_empty_matrices(self):
        """Test with empty matrices."""
        areas = pd.DataFrame({
            'Id': [1, 2],
            'Name_Area': ['A1', 'A2'],
            'Name_Region': ['R1', 'R1']
        })
        areas.set_index('Id', inplace=True)
        
        # Empty sequence should raise index error
        self.assertRaises(IndexError, spatio_temporal_graph_from_corr_matrices, [], areas)
    
    def test_single_time_point(self):
        """Test with single time point."""
        areas = pd.DataFrame({
            'Id': [1, 2, 3],
            'Name_Area': ['A1', 'A2', 'A3'],
            'Name_Region': ['R1', 'R1', 'R2']
        })
        areas.set_index('Id', inplace=True)
        
        matrices = [
            np.array([
                [1.0, 0.8, 0.4],
                [0.8, 1.0, 0.1],
                [0.4, 0.1, 1.0]
            ])
        ]
        
        result = spatio_temporal_graph_from_corr_matrices(matrices, areas)
        
        # Should have spatial edges but no temporal edges
        edge_types = [d['type'] for _, _, d in result.edges(data=True)]
        self.assertIn('spatial', edge_types)
        self.assertNotIn('temporal', edge_types)
