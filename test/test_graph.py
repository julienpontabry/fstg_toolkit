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
import pandas as pd

from fstg_toolkit.graph import RC5, subgraph_nodes, subgraph_edges, SpatioTemporalGraph, are_st_graphs_close


class RC5EnumTestCase(unittest.TestCase):
    """Test RC5 enum functionality."""
    
    def test_enum_values(self):
        """Test that all RC5 values are present."""
        expected_transitions = ['EQ', 'PP', 'PPi', 'PO', 'DC']
        actual_transitions = [t.name for t in RC5]
        
        self.assertEqual(sorted(expected_transitions), sorted(actual_transitions))
    
    def test_from_name_valid(self):
        """Test from_name with valid transition names."""
        for transition in RC5:
            result = RC5.from_name(transition.name)
            self.assertEqual(result, transition)
    
    def test_from_name_invalid(self):
        """Test from_name with invalid transition names."""
        invalid_names = ['INVALID', 'eq', 'pp', '', 'XYZ']
        
        for name in invalid_names:
            with self.assertRaises(ValueError):
                RC5.from_name(name)
    
    def test_string_representation(self):
        """Test string representation of RC5 values."""
        for transition in RC5:
            self.assertEqual(str(transition), transition.name)


class SubgraphNodesTestCase(unittest.TestCase):
    """Test subgraph_nodes function."""
    
    def setUp(self):
        """Set up a test graph."""
        self.graph = nx.Graph()
        self.graph.add_nodes_from([
            (1, dict(a=0, b=1, t=0)),
            (2, dict(a=2, b=1, t=0)),
            (3, dict(a=2, b=2, t=1)),
            (4, dict(a=2, b=1, t=1))
        ])
        self.graph.add_edges_from([(1, 2), (3, 4), (1, 4)])
    
    def test_no_conditions(self):
        """Test with no filtering conditions."""
        result = subgraph_nodes(self.graph)
        self.assertEqual(set(result.nodes), {1, 2, 3, 4})
    
    def test_single_condition(self):
        """Test with single filtering condition."""
        result = subgraph_nodes(self.graph, a=0)
        self.assertEqual(set(result.nodes), {1})
        
        result = subgraph_nodes(self.graph, b=1)
        self.assertEqual(set(result.nodes), {1, 2, 4})
        
        result = subgraph_nodes(self.graph, t=1)
        self.assertEqual(set(result.nodes), {3, 4})
    
    def test_multiple_conditions(self):
        """Test with multiple filtering conditions."""
        result = subgraph_nodes(self.graph, a=2, b=1)
        self.assertEqual(set(result.nodes), {2, 4})
        
        result = subgraph_nodes(self.graph, b=2, a=2)
        self.assertEqual(set(result.nodes), {3})
    
    def test_iterable_conditions(self):
        """Test with iterable conditions."""
        result = subgraph_nodes(self.graph, b=(1, 2), a=2)
        self.assertEqual(set(result.nodes), {2, 3, 4})
        
        result = subgraph_nodes(self.graph, t=range(0, 2))
        self.assertEqual(set(result.nodes), {1, 2, 3, 4})
    
    def test_empty_result(self):
        """Test conditions that result in empty subgraph."""
        result = subgraph_nodes(self.graph, a=5)
        self.assertEqual(len(result.nodes), 0)
        
        result = subgraph_nodes(self.graph, b=3, a=1)
        self.assertEqual(len(result.nodes), 0)


class SubgraphEdgesTestCase(unittest.TestCase):
    """Test subgraph_edges function."""
    
    def setUp(self):
        """Set up a test graph with edge attributes."""
        self.graph = nx.Graph()
        self.graph.add_nodes_from([1, 2, 3, 4])
        self.graph.add_edges_from([
            (1, 2, dict(type='spatial', correlation=0.8)),
            (2, 3, dict(type='temporal', transition='EQ')),
            (3, 4, dict(type='spatial', correlation=0.6)),
            (1, 4, dict(type='temporal', transition='PP'))
        ])
    
    def test_no_conditions(self):
        """Test with no filtering conditions."""
        result = subgraph_edges(self.graph)
        self.assertEqual(set(result.edges), {(1, 2), (2, 3), (3, 4), (1, 4)})
    
    def test_single_condition(self):
        """Test with single filtering condition."""
        result = subgraph_edges(self.graph, type='spatial')
        self.assertEqual(set(result.edges), {(1, 2), (3, 4)})
        
        result = subgraph_edges(self.graph, type='temporal')
        self.assertEqual(set(result.edges), {(2, 3), (1, 4)})
    
    def test_multiple_conditions(self):
        """Test with multiple filtering conditions."""
        result = subgraph_edges(self.graph, type='temporal', transition='EQ')
        self.assertEqual(set(result.edges), {(2, 3)})
    
    def test_iterable_conditions(self):
        """Test with iterable conditions."""
        result = subgraph_edges(self.graph, transition=['EQ', 'PP'])
        self.assertEqual(set(result.edges), {(2, 3), (1, 2), (3, 4), (1, 4)})


class SpatioTemporalGraphTestCase(unittest.TestCase):
    """Test SpatioTemporalGraph class."""
    
    def setUp(self):
        """Set up a test spatio-temporal graph."""
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from([
            (1, dict(t=0, areas={1, 2}, region='R1', internal_strength=0.8)),
            (2, dict(t=0, areas={3}, region='R2', internal_strength=1.0)),
            (3, dict(t=1, areas={1, 2, 3}, region='R1', internal_strength=0.9)),
            (4, dict(t=1, areas={4}, region='R2', internal_strength=0.7))
        ])
        self.graph.add_edges_from([
            (1, 2, dict(type='spatial', correlation=0.6)),
            (2, 1, dict(type='spatial', correlation=0.6)),
            (1, 3, dict(type='temporal', transition=RC5.PP)),
            (2, 4, dict(type='temporal', transition=RC5.PO)),
            (3, 4, dict(type='spatial', correlation=0.5))
        ])
        self.graph.graph['min_time'] = 0
        self.graph.graph['max_time'] = 1
        
        self.areas = pd.DataFrame({
            'Id_Area': [1, 2, 3, 4],
            'Name_Area': ['A1', 'A2', 'A3', 'A4'],
            'Name_Region': ['R1', 'R1', 'R2', 'R2']
        })
        self.areas.set_index('Id_Area', inplace=True)
        
        self.st_graph = SpatioTemporalGraph(self.graph, self.areas)
    
    def test_time_range(self):
        """Test time_range property."""
        self.assertEqual(self.st_graph.time_range, range(0, 2))
    
    def test_sub_method(self):
        """Test sub method with various conditions."""
        # Test time filtering
        result = self.st_graph.sub(t=0)
        self.assertEqual(set(result.nodes), {1, 2})
        
        # Test region filtering
        result = self.st_graph.sub(region='R1')
        self.assertEqual(set(result.nodes), {1, 3})
        
        # Test edge type filtering
        result = self.st_graph.sub(type='temporal')
        temporal_edges = set(result.edges)
        self.assertEqual(temporal_edges, {(1, 3), (2, 4)})
        
        # Test combined conditions
        result = self.st_graph.sub(t=1, region='R1')
        self.assertEqual(set(result.nodes), {3})
    
    def test_sub_spatial(self):
        """Test sub_spatial method."""
        result = self.st_graph.sub_spatial()
        spatial_edges = set(result.edges)
        self.assertEqual(spatial_edges, {(1, 2), (2, 1), (3, 4)})
    
    def test_sub_temporal(self):
        """Test sub_temporal method."""
        result = self.st_graph.sub_temporal()
        temporal_edges = set(result.edges)
        self.assertEqual(temporal_edges, {(1, 3), (2, 4)})
    
    def test_equality(self):
        """Test equality comparison."""
        # Test equality with same graph
        same_graph = SpatioTemporalGraph(self.graph.copy(), self.areas.copy())
        self.assertEqual(self.st_graph, same_graph)
        
        # Test inequality with different graph
        different_graph = nx.DiGraph(self.graph)
        different_graph.add_node(5, t=2, areas={5}, region='R1', internal_strength=0.5)
        different_st = SpatioTemporalGraph(different_graph, self.areas)
        self.assertNotEqual(self.st_graph, different_st)
        
        # Test inequality with different areas
        different_areas = self.areas.copy()
        different_areas.loc[1, 'Name_Area'] = 'Changed'
        different_st = SpatioTemporalGraph(self.graph.copy(), different_areas)
        self.assertNotEqual(self.st_graph, different_st)
    
    def test_string_representation(self):
        """Test string representation."""
        repr_str = str(self.st_graph)
        self.assertIn('SpatioTemporalGraph', repr_str)
        self.assertIn('#areas=4', repr_str)
        self.assertIn('#regions=2', repr_str)
        self.assertIn('#nodes=4', repr_str)


class AreSTGraphsCloseTestCase(unittest.TestCase):
    """Test are_st_graphs_close function."""
    
    def setUp(self):
        """Set up test graphs."""
        # Create base graph
        self.graph1 = nx.DiGraph()
        self.graph1.add_nodes_from([
            (1, dict(t=0, areas={1, 2}, region='R1', internal_strength=0.8)),
            (2, dict(t=1, areas={1, 2}, region='R1', internal_strength=0.85))
        ])
        self.graph1.add_edge(1, 2, type='temporal', transition=RC5.EQ)
        self.graph1.graph['min_time'] = 0
        self.graph1.graph['max_time'] = 1
        
        self.areas = pd.DataFrame({
            'Id_Area': [1, 2],
            'Name_Area': ['A1', 'A2'],
            'Name_Region': ['R1', 'R1']
        })
        self.areas.set_index('Id_Area', inplace=True)
        
        self.st_graph1 = SpatioTemporalGraph(self.graph1, self.areas)
    
    def test_identical_graphs(self):
        """Test with identical graphs."""
        graph2 = self.graph1.copy()
        areas2 = self.areas.copy()
        st_graph2 = SpatioTemporalGraph(graph2, areas2)
        
        self.assertTrue(are_st_graphs_close(self.st_graph1, st_graph2))
    
    def test_numerical_tolerance(self):
        """Test numerical tolerance."""
        # Create graph with slightly different numerical values
        graph2 = nx.DiGraph()
        graph2.add_nodes_from([
            (1, dict(t=0, areas={1, 2}, region='R1', internal_strength=0.8000000001)),
            (2, dict(t=1, areas={1, 2}, region='R1', internal_strength=0.8499999999))
        ])
        graph2.add_edge(1, 2, type='temporal', transition=RC5.EQ)
        graph2.graph['min_time'] = 0
        graph2.graph['max_time'] = 1
        
        st_graph2 = SpatioTemporalGraph(graph2, self.areas.copy())
        
        # Should be considered equal within tolerance
        self.assertTrue(are_st_graphs_close(self.st_graph1, st_graph2))
    
    def test_different_structure(self):
        """Test with different graph structure."""
        graph2 = nx.DiGraph()
        graph2.add_nodes_from([
            (1, dict(t=0, areas={1}, region='R1', internal_strength=0.8)),
            (2, dict(t=1, areas={1, 2}, region='R1', internal_strength=0.85))
        ])
        graph2.add_edge(1, 2, type='temporal', transition=RC5.PP)
        graph2.graph['min_time'] = 0
        graph2.graph['max_time'] = 1
        
        st_graph2 = SpatioTemporalGraph(graph2, self.areas.copy())
        
        # Should not be equal due to different structure
        self.assertFalse(are_st_graphs_close(self.st_graph1, st_graph2))
    
    def test_different_areas(self):
        """Test with different areas DataFrame."""
        areas2 = self.areas.copy()
        areas2.loc[1, 'Name_Area'] = 'Different'
        st_graph2 = SpatioTemporalGraph(self.graph1.copy(), areas2)
        
        # Should not be equal due to different areas
        self.assertFalse(are_st_graphs_close(self.st_graph1, st_graph2))
    
    def test_outside_tolerance(self):
        """Test values outside numerical tolerance."""
        graph2 = nx.DiGraph()
        graph2.add_nodes_from([
            (1, dict(t=0, areas={1, 2}, region='R1', internal_strength=0.9)),  # Different enough
            (2, dict(t=1, areas={1, 2}, region='R1', internal_strength=0.85))
        ])
        graph2.add_edge(1, 2, type='temporal', transition=RC5.EQ)
        graph2.graph['min_time'] = 0
        graph2.graph['max_time'] = 1
        
        st_graph2 = SpatioTemporalGraph(graph2, self.areas.copy())
        
        # Should not be equal due to values outside tolerance
        self.assertFalse(are_st_graphs_close(self.st_graph1, st_graph2))
