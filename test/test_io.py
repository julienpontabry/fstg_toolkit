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

import tempfile
import unittest
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from fstg_toolkit.graph import SpatioTemporalGraph, RC5
from fstg_toolkit.io import DataLoader, DataSaver, load_spatio_temporal_graph, \
    save_spatio_temporal_graph, load_metrics, save_metrics


class DataLoaderTestCase(unittest.TestCase):
    """Test DataLoader class functionality."""
    
    def setUp(self):
        """Set up test data."""
        # Create a temporary zip file with test data
        self.temp_dir = tempfile.mkdtemp()
        self.zip_path = Path(self.temp_dir) / 'test_data.zip'
        
        # Create test data
        self.areas = pd.DataFrame({
            'Id_Area': [1, 2, 3, 4],
            'Name_Area': ['A1', 'A2', 'A3', 'A4'],
            'Name_Region': ['R1', 'R1', 'R2', 'R2']
        }).set_index('Id_Area')
        
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from([
            (1, dict(t=0, areas={1, 2}, region='R1', internal_strength=0.8)),
            (2, dict(t=1, areas={1, 2, 3, 4}, region='R1', internal_strength=0.9))
        ])
        self.graph.add_edge(1, 2, type='temporal', transition=RC5.PP)
        self.graph.graph['min_time'] = 0
        self.graph.graph['max_time'] = 1
        
        self.matrix = np.array([[1.0, 0.8], [0.8, 1.0]])
        
        # Save test data to zip file
        self._create_test_zip_file()
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def _create_test_zip_file(self):
        """Create a test zip file with sample data."""
        import zipfile
        
        with zipfile.ZipFile(self.zip_path, 'w') as zf:
            # Save areas
            with zf.open('areas.csv', 'w') as f:
                self.areas.to_csv(f)
            
            # Save graph
            import json
            from fstg_toolkit.io import _SpatioTemporalGraphEncoder
            graph_dict = nx.json_graph.node_link_data(self.graph, edges='edges')
            graph_json = json.dumps(graph_dict, cls=_SpatioTemporalGraphEncoder)
            zf.writestr('test_graph.json', graph_json)

            # Save frequent patterns style (use same graph; not important for those tests)
            zf.writestr('test_graph/motifs_enriched_t.json', graph_json)
            
            # Save matrix
            with zf.open('test_matrix.npy', 'w') as f:
                np.save(f, self.matrix)
    
    def test_load_areas(self):
        """Test loading areas DataFrame."""
        loader = DataLoader(self.zip_path)
        areas = loader.load_areas()
        
        self.assertIsNotNone(areas)
        self.assertEqual(len(areas), 4)
        self.assertListEqual(list(areas.columns), ['Name_Area', 'Name_Region'])
        self.assertEqual(areas.index.name, 'Id_Area')
    
    def test_load_graphs(self):
        """Test loading graphs."""
        loader = DataLoader(self.zip_path)
        graphs = loader.load_graphs(self.areas)
        
        self.assertEqual(len(graphs), 1)
        self.assertIn('test_graph', graphs)
        
        graph = graphs['test_graph']
        self.assertIsInstance(graph, SpatioTemporalGraph)
        self.assertEqual(len(graph.nodes), 2)
        self.assertEqual(len(graph.edges), 1)
    
    def test_load_matrices(self):
        """Test loading matrices."""
        loader = DataLoader(self.zip_path)
        matrices = loader.load_matrices()
        
        self.assertEqual(len(matrices), 1)
        self.assertIn('test_matrix', matrices)
        
        matrix = matrices['test_matrix']
        np.testing.assert_array_equal(matrix, self.matrix)
    
    def test_lazy_load_graphs(self):
        """Test lazy loading of graph filenames."""
        loader = DataLoader(self.zip_path)
        filenames = loader.lazy_load_graphs()
        
        self.assertEqual(len(filenames), 1)
        self.assertEqual(filenames[0], 'test_graph.json')

    def test_lazy_load_frequent_patterns(self):
        """Test lazy loading of frequent patterns."""
        loader = DataLoader(self.zip_path)
        filenames = loader.lazy_load_frequent_patterns()

        self.assertEqual(len(filenames), 1)
        self.assertEqual(filenames[0], 'test_graph/motifs_enriched_t.json')
    
    def test_lazy_load_matrices(self):
        """Test lazy loading of matrix filenames."""
        loader = DataLoader(self.zip_path)
        filenames = loader.lazy_load_matrices()
        
        self.assertEqual(len(filenames), 1)
        self.assertEqual(filenames[0], 'test_matrix.npy')
    
    def test_load_graph(self):
        """Test loading single graph."""
        loader = DataLoader(self.zip_path)
        graph = loader.load_graph(self.areas, 'test_graph.json')
        
        self.assertIsNotNone(graph)
        self.assertIsInstance(graph, SpatioTemporalGraph)
        self.assertEqual(len(graph.nodes), 2)
    
    def test_load_matrix(self):
        """Test loading single matrix."""
        loader = DataLoader(self.zip_path)
        matrix = loader.load_matrix('test_matrix.npy')
        
        self.assertIsNotNone(matrix)
        np.testing.assert_array_equal(matrix, self.matrix)
    
    def test_load_all(self):
        """Test loading all data."""
        loader = DataLoader(self.zip_path)
        areas, graphs, matrices = loader.load()
        
        self.assertIsNotNone(areas)
        self.assertEqual(len(graphs), 1)
        self.assertEqual(len(matrices), 1)
    
    def test_lazy_load(self):
        """Test lazy loading."""
        loader = DataLoader(self.zip_path)
        areas, graph_files, matrix_files = loader.lazy_load()
        
        self.assertIsNotNone(areas)
        self.assertEqual(len(graph_files), 1)
        self.assertEqual(len(matrix_files), 1)
    
    def test_nonexistent_file(self):
        """Test with nonexistent file."""
        nonexistent_path = Path(self.temp_dir) / 'nonexistent.zip'
        
        with self.assertRaises(FileNotFoundError):
            DataLoader(nonexistent_path)
    
    def test_missing_areas_file(self):
        """Test with missing areas file."""
        # Create zip without areas.csv
        no_areas_path = Path(self.temp_dir) / 'no_areas.zip'
        import zipfile
        
        with zipfile.ZipFile(no_areas_path, 'w') as zf:
            zf.writestr('test.json', '{}')
        
        loader = DataLoader(no_areas_path)

        with self.assertRaisesRegex(KeyError, "There is no item named 'areas.csv' in the archive"):
            loader.load_areas()


class DataSaverTestCase(unittest.TestCase):
    """Test DataSaver class functionality."""
    
    def setUp(self):
        """Set up test data."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test data
        self.areas = pd.DataFrame({
            'Id_Area': [1, 2, 3],
            'Name_Area': ['A1', 'A2', 'A3'],
            'Name_Region': ['R1', 'R1', 'R2']
        }).set_index('Id_Area')
        
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from([
            (1, dict(t=0, areas={1, 2}, region='R1', internal_strength=0.8)),
            (2, dict(t=1, areas={1, 2, 3}, region='R1', internal_strength=0.9))
        ])
        self.graph.add_edge(1, 2, type='temporal', transition=RC5.PP)
        self.graph.graph['min_time'] = 0
        self.graph.graph['max_time'] = 1
        
        self.st_graph = SpatioTemporalGraph(self.graph, self.areas)
        
        self.matrix = np.array([[1.0, 0.8, 0.2], [0.8, 1.0, 0.1], [0.2, 0.1, 1.0]])
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_save_and_load_roundtrip(self):
        """Test save and load roundtrip."""
        save_path = Path(self.temp_dir) / 'test_save.zip'
        
        # Create saver and add data
        saver = DataSaver()
        saver.add(self.areas)
        saver.add({'test_graph': self.st_graph})
        saver.add({'test_matrix': self.matrix})
        
        # Save data
        saver.save(save_path)
        
        # Load data back
        loader = DataLoader(save_path)
        loaded_areas = loader.load_areas()
        loaded_graphs = loader.load_graphs(loaded_areas)
        loaded_matrices = loader.load_matrices()
        
        # Verify areas
        pd.testing.assert_frame_equal(loaded_areas, self.areas)
        
        # Verify graph
        self.assertEqual(len(loaded_graphs), 1)
        loaded_graph = loaded_graphs['test_graph']
        self.assertEqual(len(loaded_graph.nodes), len(self.st_graph.nodes))
        self.assertEqual(len(loaded_graph.edges), len(self.st_graph.edges))
        
        # Verify matrix
        self.assertEqual(len(loaded_matrices), 1)
        np.testing.assert_array_equal(loaded_matrices['test_matrix'], self.matrix)
    
    def test_multiple_elements(self):
        """Test saving multiple elements."""
        save_path = Path(self.temp_dir) / 'multiple.zip'
        
        # Create multiple graphs and matrices
        graphs = {
            'graph1': self.st_graph,
            'graph2': SpatioTemporalGraph(self.graph.copy(), self.areas.copy())
        }
        
        matrices = {
            'matrix1': self.matrix,
            'matrix2': np.eye(3)
        }
        
        saver = DataSaver()
        saver.add(self.areas)
        saver.add(graphs)
        saver.add(matrices)
        
        saver.save(save_path)
        
        # Verify file was created
        self.assertTrue(save_path.exists())
        
        # Load and verify
        loader = DataLoader(save_path)
        loaded_areas = loader.load_areas()
        loaded_graphs = loader.load_graphs(loaded_areas)
        loaded_matrices = loader.load_matrices()
        
        self.assertEqual(len(loaded_graphs), 2)
        self.assertEqual(len(loaded_matrices), 2)
    
    def test_clear_and_add(self):
        """Test clear and add functionality."""
        save_path = Path(self.temp_dir) / 'clear_test.zip'
        
        saver = DataSaver()
        saver.add(self.areas)
        self.assertEqual(len(saver.elements), 1)
        
        saver.clear()
        self.assertEqual(len(saver.elements), 0)
        
        saver.add({'test': self.st_graph})
        self.assertEqual(len(saver.elements), 1)
        
        saver.save(save_path)
        self.assertTrue(save_path.exists())


class LoadSaveSpatioTemporalGraphTestCase(unittest.TestCase):
    """Test load/save spatio-temporal graph functions."""
    
    def setUp(self):
        """Set up test data."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test graph
        self.areas = pd.DataFrame({
            'Id_Area': [1, 2, 3, 4],
            'Name_Area': ['A1', 'A2', 'A3', 'A4'],
            'Name_Region': ['R1', 'R1', 'R2', 'R2']
        })
        self.areas.set_index('Id_Area', inplace=True)
        
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from([
            (1, dict(t=0, areas={1, 2}, region='R1', internal_strength=0.8, efficiency=1.0)),
            (2, dict(t=0, areas={3, 4}, region='R2', internal_strength=0.7, efficiency=1.0)),
            (3, dict(t=1, areas={1, 2, 3}, region='R1', internal_strength=0.85, efficiency=0.9)),
            (4, dict(t=1, areas={4}, region='R2', internal_strength=1.0, efficiency=1.0))
        ])
        self.graph.add_edges_from([
            (1, 2, dict(type='spatial', correlation=0.6, t=0)),
            (2, 1, dict(type='spatial', correlation=0.6, t=0)),
            (1, 3, dict(type='temporal', transition=RC5.PP)),
            (2, 4, dict(type='temporal', transition=RC5.PO)),
            (3, 4, dict(type='spatial', correlation=0.5, t=1)),
            (4, 3, dict(type='spatial', correlation=0.5, t=1))
        ])
        self.graph.graph['min_time'] = 0
        self.graph.graph['max_time'] = 1
        
        self.st_graph = SpatioTemporalGraph(self.graph, self.areas)
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_save_and_load_roundtrip(self):
        """Test save and load roundtrip preserves graph structure."""
        save_path = Path(self.temp_dir) / 'graph_test.zip'
        
        # Save graph
        save_spatio_temporal_graph(self.st_graph, save_path)
        
        # Load graph back
        loaded_graph = load_spatio_temporal_graph(save_path)
        
        # Verify structure
        self.assertEqual(len(loaded_graph.nodes), len(self.st_graph.nodes))
        self.assertEqual(len(loaded_graph.edges), len(self.st_graph.edges))
        
        # Verify node attributes
        for node in self.st_graph.nodes():
            original_attrs = self.st_graph.nodes[node]
            loaded_attrs = loaded_graph.nodes[node]
            
            for key, value in original_attrs.items():
                if isinstance(value, set):
                    self.assertSetEqual(value, loaded_attrs[key])
                else:
                    self.assertEqual(value, loaded_attrs[key])
        
        # Verify edge attributes
        for u, v, attrs in self.st_graph.edges(data=True):
            loaded_attrs = loaded_graph.edges[u, v]
            
            for key, value in attrs.items():
                if key == 'transition':
                    self.assertEqual(value, loaded_attrs[key])
                else:
                    self.assertEqual(value, loaded_attrs[key])
        
        # Verify areas
        pd.testing.assert_frame_equal(loaded_graph.areas, self.st_graph.areas)
    
    def test_load_nonexistent_file(self):
        """Test loading from nonexistent file."""
        nonexistent_path = Path(self.temp_dir) / 'nonexistent.zip'
        
        with self.assertRaises(FileNotFoundError):
            load_spatio_temporal_graph(nonexistent_path)
    
    def test_load_empty_archive(self):
        """Test loading from archive with no graphs."""
        empty_path = Path(self.temp_dir) / 'empty.zip'
        
        import zipfile
        with zipfile.ZipFile(empty_path, 'w') as zf:
            zf.writestr('areas.csv', self.areas.to_csv())
        
        with self.assertRaises(RuntimeError):
            load_spatio_temporal_graph(empty_path)


class MetricsIOTestCase(unittest.TestCase):
    """Test metrics save/load functionality."""
    
    def setUp(self):
        """Set up test data."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test metrics DataFrame
        self.simple_metrics = pd.DataFrame({
            'metric1': [0.1, 0.2, 0.3],
            'metric2': [0.4, 0.5, 0.6]
        }, index=pd.Index(['subject1', 'subject2', 'subject3'], name='id'))
        
        # Create multi-index metrics
        self.multi_index_metrics = pd.DataFrame(
            np.random.rand(4, 3),
            index=pd.MultiIndex.from_tuples([
                ('A', 'x'), ('A', 'y'), ('B', 'x'), ('B', 'y')
            ], names=['factor1', 'factor2']),
            columns=pd.MultiIndex.from_tuples([
                ('metric1', 'value'), ('metric1', 'pvalue'), ('metric2', 'value')
            ])
        )
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_simple_metrics_roundtrip(self):
        """Test save/load roundtrip with simple metrics."""
        save_path = Path(self.temp_dir) / 'simple_metrics.csv'
        
        # Save metrics
        save_metrics(save_path, self.simple_metrics)
        
        # Load metrics back
        loaded_metrics = load_metrics(save_path)
        
        # Verify
        pd.testing.assert_frame_equal(loaded_metrics, self.simple_metrics)
    
    def test_multi_index_metrics_roundtrip(self):
        """Test save/load roundtrip with multi-index metrics."""
        save_path = Path(self.temp_dir) / 'multi_metrics.csv'
        
        # Save metrics
        save_metrics(save_path, self.multi_index_metrics)
        
        # Load metrics back
        loaded_metrics = load_metrics(save_path)
        
        # Verify structure
        self.assertIsInstance(loaded_metrics.index, pd.MultiIndex)
        self.assertIsInstance(loaded_metrics.columns, pd.MultiIndex)
        
        # Verify values
        pd.testing.assert_frame_equal(loaded_metrics, self.multi_index_metrics)
    
    def test_mixed_index_metrics(self):
        """Test metrics with only column multi-index."""
        mixed_metrics = pd.DataFrame(
            np.random.rand(3, 2),
            index=pd.Index(['A', 'B', 'C'], name='id'),
            columns=pd.MultiIndex.from_tuples([
                ('metric1', 'value'), ('metric2', 'value')
            ])
        )
        
        save_path = Path(self.temp_dir) / 'mixed_metrics.csv'
        
        # Save and load
        save_metrics(save_path, mixed_metrics)
        loaded_metrics = load_metrics(save_path)
        
        # Verify
        pd.testing.assert_frame_equal(loaded_metrics, mixed_metrics)
    
    def test_nonexistent_metrics_file(self):
        """Test loading from nonexistent file."""
        nonexistent_path = Path(self.temp_dir) / 'nonexistent.csv'
        
        with self.assertRaises(FileNotFoundError):
            load_metrics(nonexistent_path)
