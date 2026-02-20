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

from fstg_toolkit.graph import SpatioTemporalGraph, RC5
from fstg_toolkit.io import load_spatio_temporal_graph
from fstg_toolkit.metrics import (
    MetricsRegistry, get_metrics_registry, calculate_spatial_metrics,
    calculate_temporal_metrics, metric,
    average_degree, assortativity, clustering, global_efficiency,
    density, modularity, transitions_distribution, reorg_rate,
    burstiness, memory
)
from test_common import graph_path


class MetricsCalculationTestCase(unittest.TestCase):
    def setUp(self):
        self.graph = load_spatio_temporal_graph(graph_path)

    def test_calculate_metrics(self):
        df = pd.DataFrame.from_records(calculate_spatial_metrics(self.graph))
        
        registry = list(get_metrics_registry('local'))
        for metric, _ in registry:
            self.assertIn(metric, df.columns)

        self.assertEqual(len(df), len(self.graph.time_range))

    def test_calculate_temporal_metrics(self):
        df = pd.DataFrame.from_records(calculate_temporal_metrics(self.graph))

        registry = list(get_metrics_registry('global'))
        for metric, _ in registry:
            self.assertIn(metric, df.columns)

        self.assertEqual(len(df), 1)


class MetricsRegistryTestCase(unittest.TestCase):
    """Test MetricsRegistry functionality."""
    
    def setUp(self):
        """Set up test registry."""
        self.registry = MetricsRegistry()
        self.test_func = lambda x: 42
    
    def test_add_and_iterate(self):
        """Test adding and iterating over metrics."""
        self.registry.add('test_metric', self.test_func)
        
        metrics = list(self.registry)
        self.assertEqual(len(metrics), 1)
        self.assertEqual(metrics[0][0], 'test_metric')
        self.assertEqual(metrics[0][1], self.test_func)
    
    def test_remove(self):
        """Test removing metrics."""
        self.registry.add('test_metric', self.test_func)
        self.assertEqual(len(list(self.registry)), 1)
        
        self.registry.remove('test_metric')
        self.assertEqual(len(list(self.registry)), 0)
    
    def test_get_metrics_registry(self):
        """Test getting named registries."""
        registry1 = get_metrics_registry('test1')
        registry2 = get_metrics_registry('test1')
        registry3 = get_metrics_registry('test2')
        
        # Same name should return same registry
        self.assertIs(registry1, registry2)
        # Different names should return different registries
        self.assertIsNot(registry1, registry3)


class MetricDecoratorTestCase(unittest.TestCase):
    """Test metric decorator functionality."""
    
    def test_metric_decorator(self):
        """Test that metric decorator adds function to registry."""
        test_registry = get_metrics_registry('test_decorator')
        initial_count = len(list(test_registry))
        
        @metric('test_decorator', 'test_metric')
        def test_func(graph):
            return 42
        
        # Check that function was added to registry
        metrics = list(test_registry)
        self.assertEqual(len(metrics), initial_count + 1)
        self.assertEqual(metrics[-1][0], 'test_metric')
        self.assertEqual(metrics[-1][1], test_func)


class IndividualMetricFunctionsTestCase(unittest.TestCase):
    """Test individual metric calculation functions."""
    
    def setUp(self):
        """Set up test graphs."""
        # Create a simple test graph
        self.simple_graph = nx.Graph()
        self.simple_graph.add_edges_from([
            (1, 2), (2, 3), (3, 4), (4, 1),  # Square
            (1, 3)  # Diagonal
        ])
        
        # Create graph with known properties
        self.known_graph = nx.Graph()
        self.known_graph.add_edges_from([
            (1, 2), (1, 3), (1, 4),  # Star-like
            (2, 3), (3, 4)
        ])
        
        # Create areas DataFrame
        self.areas = pd.DataFrame({
            'Id_Area': [1, 2, 3, 4],
            'Name_Area': ['A1', 'A2', 'A3', 'A4'],
            'Name_Region': ['R1', 'R1', 'R2', 'R2']
        })
        self.areas.set_index('Id_Area', inplace=True)
        
        # Create SpatioTemporalGraph
        self.st_graph = SpatioTemporalGraph(self.known_graph, self.areas)
    
    def test_average_degree(self):
        """Test average degree calculation."""
        # Simple graph: degrees are [3, 2, 3, 2], average = 2.5
        result = average_degree(self.st_graph)
        self.assertAlmostEqual(result, 0.5714285714285714)
    
    def test_assortativity(self):
        """Test assortativity calculation."""
        # This tests that the function runs without error
        # Exact value depends on graph structure
        result = assortativity(self.st_graph)
        self.assertIsInstance(result, float)
        self.assertTrue(-1 <= result <= 1)
    
    def test_clustering(self):
        """Test clustering coefficient calculation."""
        result = clustering(self.st_graph)
        self.assertIsInstance(result, float)
        self.assertTrue(0 <= result <= 1)
    
    def test_global_efficiency(self):
        """Test global efficiency calculation."""
        result = global_efficiency(self.st_graph)
        self.assertIsInstance(result, float)
        self.assertTrue(0 <= result <= 1)
    
    def test_density(self):
        """Test density calculation."""
        # Known graph has 5 edges out of 6 possible
        expected = 5 / 6
        result = density(self.st_graph)
        self.assertAlmostEqual(result, expected)
    
    def test_modularity(self):
        """Test modularity calculation."""
        result = modularity(self.st_graph)
        self.assertIsInstance(result, float)
        self.assertTrue(0 <= result <= 1)
    
    def test_empty_graph(self):
        """Test metrics on empty graph."""
        empty_graph = nx.Graph()
        st_empty = SpatioTemporalGraph(empty_graph, self.areas.iloc[:0])
        
        # These should handle empty graphs gracefully
        self.assertTrue(np.isnan(average_degree(st_empty)))
        self.assertEqual(density(st_empty), 0.0)

        with self.assertRaises(ZeroDivisionError):
            clustering(st_empty)
            modularity(st_empty)
    
    def test_single_node_graph(self):
        """Test metrics on single node graph."""
        single_graph = nx.Graph()
        single_graph.add_node(1)
        areas_single = self.areas.iloc[[0]]
        st_single = SpatioTemporalGraph(single_graph, areas_single)
        
        self.assertEqual(average_degree(st_single), 1.0)
        self.assertEqual(clustering(st_single), 0.0)
        self.assertEqual(density(st_single), 0.0)


class TemporalMetricsTestCase(unittest.TestCase):
    """Test temporal metric functions."""
    
    def setUp(self):
        """Set up test spatio-temporal graph with temporal edges."""
        # Create a graph with various temporal transitions
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from([
            (1, dict(t=0, areas={1, 2}, region='R1', internal_strength=0.8)),
            (2, dict(t=0, areas={3}, region='R1', internal_strength=1.0)),
            (3, dict(t=1, areas={1}, region='R1', internal_strength=0.7)),
            (4, dict(t=1, areas={2, 3}, region='R1', internal_strength=0.9)),
            (5, dict(t=2, areas={1, 2, 3}, region='R1', internal_strength=0.85))
        ])
        
        # Add temporal edges with different transitions
        self.graph.add_edge(1, 3, type='temporal', transition=RC5.PPi)  # Split
        self.graph.add_edge(1, 4, type='temporal', transition=RC5.PP)   # Proper part
        self.graph.add_edge(2, 4, type='temporal', transition=RC5.PP)   # Proper part
        self.graph.add_edge(3, 5, type='temporal', transition=RC5.PP)   # Proper part
        self.graph.add_edge(4, 5, type='temporal', transition=RC5.PP)   # Proper part
        
        self.graph.graph['min_time'] = 0
        self.graph.graph['max_time'] = 2
        
        self.areas = pd.DataFrame({
            'Id_Area': [1, 2, 3],
            'Name_Area': ['A1', 'A2', 'A3'],
            'Name_Region': ['R1', 'R1', 'R1']
        })
        self.areas.set_index('Id_Area', inplace=True)
        
        self.st_graph = SpatioTemporalGraph(self.graph, self.areas)
    
    def test_transitions_distribution(self):
        """Test transitions distribution calculation."""
        result = transitions_distribution(self.st_graph)
        
        self.assertIsInstance(result, dict)
        self.assertIn(RC5.PP, result)
        self.assertIn(RC5.PPi, result)
        self.assertNotIn(RC5.DC, result)  # DC transitions are excluded
        
        # Should have 4 PP and 1 PPi transitions
        self.assertEqual(result[RC5.PP], 4)
        self.assertEqual(result[RC5.PPi], 1)
    
    def test_reorg_rate(self):
        """Test reorganization rate calculation."""
        result = reorg_rate(self.st_graph)
        
        # 5 temporal edges total, 5 are non-EQ (all PP or PPi)
        expected = 5 / 5  # 1.0
        self.assertAlmostEqual(result, expected)
    
    def test_burstiness(self):
        """Test burstiness calculation."""
        result = burstiness(self.st_graph)
        
        # This should return a value between -1 and 1
        self.assertTrue(-1 <= result <= 1)
    
    def test_memory(self):
        """Test memory calculation."""
        result = memory(self.st_graph)
        
        # This should return a value between -1 and 1
        self.assertTrue(-1 <= result <= 1)
    
    def test_burstiness_edge_cases(self):
        """Test burstiness with edge cases."""
        # Test with no events (should return -1)
        empty_graph = nx.DiGraph()
        empty_graph.add_nodes_from([
            (1, dict(t=0, areas={1}, region='R1', internal_strength=1.0)),
            (2, dict(t=1, areas={1}, region='R1', internal_strength=1.0))
        ])
        empty_graph.add_edge(1, 2, type='temporal', transition=RC5.EQ)
        empty_graph.graph['min_time'] = 0
        empty_graph.graph['max_time'] = 1
        
        st_empty = SpatioTemporalGraph(empty_graph, self.areas.iloc[[0]])
        result = burstiness(st_empty)
        self.assertEqual(result, -1)
        
        # Test with not enough events (should return 0)
        few_graph = nx.DiGraph()
        few_graph.add_nodes_from([
            (1, dict(t=0, areas={1}, region='R1', internal_strength=1.0)),
            (2, dict(t=1, areas={1, 2}, region='R1', internal_strength=1.0)),
            (3, dict(t=2, areas={1}, region='R1', internal_strength=1.0))
        ])
        few_graph.add_edge(1, 2, type='temporal', transition=RC5.PP)
        few_graph.add_edge(2, 3, type='temporal', transition=RC5.PPi)
        few_graph.graph['min_time'] = 0
        few_graph.graph['max_time'] = 2
        
        st_few = SpatioTemporalGraph(few_graph, self.areas)
        result = burstiness(st_few)
        self.assertEqual(result, 0)


class CalculateMetricsTestCase(unittest.TestCase):
    """Test metrics calculation functions."""
    
    def setUp(self):
        """Set up test spatio-temporal graph."""
        # Create a graph with spatial structure at multiple time points
        self.graph = nx.DiGraph()
        
        # Time 0
        self.graph.add_node(1, t=0, areas={1, 2}, region='R1', internal_strength=0.8, efficiency=1.0)
        self.graph.add_node(2, t=0, areas={3, 4}, region='R2', internal_strength=0.7, efficiency=1.0)
        self.graph.add_edge(1, 2, type='spatial', correlation=0.6, t=0)
        self.graph.add_edge(2, 1, type='spatial', correlation=0.6, t=0)
        
        # Time 1
        self.graph.add_node(3, t=1, areas={1, 2, 3}, region='R1', internal_strength=0.85, efficiency=0.9)
        self.graph.add_node(4, t=1, areas={4}, region='R2', internal_strength=1.0, efficiency=1.0)
        self.graph.add_edge(3, 4, type='spatial', correlation=0.5, t=1)
        self.graph.add_edge(4, 3, type='spatial', correlation=0.5, t=1)
        
        # Temporal edges
        self.graph.add_edge(1, 3, type='temporal', transition=RC5.PP)
        self.graph.add_edge(2, 4, type='temporal', transition=RC5.PO)
        
        self.graph.graph['min_time'] = 0
        self.graph.graph['max_time'] = 1
        
        self.areas = pd.DataFrame({
            'Id_Area': [1, 2, 3, 4],
            'Name_Area': ['A1', 'A2', 'A3', 'A4'],
            'Name_Region': ['R1', 'R1', 'R2', 'R2']
        })
        self.areas.set_index('Id_Area', inplace=True)
        
        self.st_graph = SpatioTemporalGraph(self.graph, self.areas)
    
    def test_calculate_spatial_metrics(self):
        """Test spatial metrics calculation."""
        records = calculate_spatial_metrics(self.st_graph)
        
        # Should have one record per time point
        self.assertEqual(len(records), 2)
        
        # Each record should have Time field
        for record in records:
            self.assertIn('Time', record)
        
        # Check that we have the expected metrics
        registry = get_metrics_registry('local')
        for metric_name, _ in registry:
            for record in records:
                self.assertIn(metric_name, record)
    
    def test_calculate_temporal_metrics(self):
        """Test temporal metrics calculation."""
        records = calculate_temporal_metrics(self.st_graph)
        
        # Should have one record for the whole graph
        self.assertEqual(len(records), 1)
        
        # Check that we have the expected metrics
        registry = get_metrics_registry('global')
        for metric_name, _ in registry:
            self.assertIn(metric_name, records[0])
    
    def test_spatial_metrics_time_values(self):
        """Test that spatial metrics have correct time values."""
        records = calculate_spatial_metrics(self.st_graph)
        
        time_values = [record['Time'] for record in records]
        self.assertEqual(sorted(time_values), [0, 1])


class EdgeCaseMetricsTestCase(unittest.TestCase):
    """Test metrics with edge case graphs."""
    
    def setUp(self):
        """Set up areas DataFrame."""
        self.areas = pd.DataFrame({
            'Id_Area': [1, 2, 3],
            'Name_Area': ['A1', 'A2', 'A3'],
            'Name_Region': ['R1', 'R1', 'R2']
        })
        self.areas.set_index('Id_Area', inplace=True)
    
    def test_empty_graph(self):
        """Test metrics on completely empty graph."""
        empty_graph = nx.DiGraph()
        empty_graph.graph['min_time'] = 0
        empty_graph.graph['max_time'] = 0
        
        st_empty = SpatioTemporalGraph(empty_graph, self.areas.iloc[:0])
        
        # Spatial metrics should handle empty graph
        with self.assertRaises(ZeroDivisionError):
            calculate_spatial_metrics(st_empty)
        
        # Temporal metrics should handle empty graph
        with self.assertRaises(ZeroDivisionError):
            calculate_temporal_metrics(st_empty)
    
    def test_disconnected_graph(self):
        """Test metrics on disconnected graph."""
        graph = nx.DiGraph()
        graph.add_nodes_from([
            (1, dict(t=0, areas={1}, region='R1', internal_strength=1.0)),
            (2, dict(t=0, areas={2}, region='R1', internal_strength=1.0)),
            (3, dict(t=0, areas={3}, region='R2', internal_strength=1.0))
        ])
        # No edges - completely disconnected
        graph.graph['min_time'] = 0
        graph.graph['max_time'] = 0
        
        st_disconnected = SpatioTemporalGraph(graph, self.areas)
        local_metrics = calculate_spatial_metrics(st_disconnected)

        expected_metrics = [
            {'Time': 0, 'Average degree': 3.0, 'Assortativity': float('nan'),
             'Clustering coefficient': 0.0, 'Global efficiency': 0.0,
             'Density': 0, 'Modularity': 0.0}]
        for key in local_metrics[0]:
            if not np.isnan(local_metrics[0][key]):
                self.assertEqual(local_metrics[0][key], expected_metrics[0][key])
