# Copyright 2025 ICube (University of Strasbourg - CNRS)
# author: Julien PONTABRY (ICube)

import tempfile
import unittest
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from fstg_toolkit import (
    spatio_temporal_graph_from_corr_matrices,
    CorrelationMatrixSequenceSimulator,
    SpatioTemporalGraphSimulator, generate_pattern,
    load_spatio_temporal_graph, save_spatio_temporal_graph
)
from fstg_toolkit.graph import SpatioTemporalGraph
from fstg_toolkit.io import load_metrics
from fstg_toolkit.metrics import calculate_spatial_metrics, calculate_temporal_metrics


class FullPipelineIntegrationTestCase(unittest.TestCase):
    """Test the complete processing pipeline from raw data to metrics."""
    
    def setUp(self):
        """Set up test data."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test areas description
        self.areas = pd.DataFrame({
            'Id_Area': [1, 2, 3, 4, 5, 6],
            'Name_Area': ['A1', 'A2', 'A3', 'A4', 'A5', 'A6'],
            'Name_Region': ['R1', 'R1', 'R1', 'R2', 'R2', 'R2']
        })
        self.areas.set_index('Id_Area', inplace=True)
        
        # Create test correlation matrices (3 time points)
        self.matrices = [
            np.array([
                [1.0, 0.8, 0.7, 0.2, 0.1, 0.0],
                [0.8, 1.0, 0.9, 0.3, 0.2, 0.1],
                [0.7, 0.9, 1.0, 0.4, 0.3, 0.2],
                [0.2, 0.3, 0.4, 1.0, 0.8, 0.7],
                [0.1, 0.2, 0.3, 0.8, 1.0, 0.9],
                [0.0, 0.1, 0.2, 0.7, 0.9, 1.0]
            ]),
            np.array([
                [1.0, 0.9, 0.8, 0.3, 0.2, 0.1],
                [0.9, 1.0, 0.95, 0.4, 0.3, 0.2],
                [0.8, 0.95, 1.0, 0.5, 0.4, 0.3],
                [0.3, 0.4, 0.5, 1.0, 0.7, 0.6],
                [0.2, 0.3, 0.4, 0.7, 1.0, 0.8],
                [0.1, 0.2, 0.3, 0.6, 0.8, 1.0]
            ]),
            np.array([
                [1.0, 0.7, 0.6, 0.1, 0.4, 0.1],
                [0.7, 1.0, 0.8, 0.2, 0.1, 0.2],
                [0.6, 0.8, 1.0, 0.3, 0.2, 0.3],
                [0.1, 0.2, 0.3, 1.0, 0.9, 0.8],
                [-0.4, 0.1, 0.2, 0.9, 1.0, 0.95],
                [0.1, 0.2, 0.3, 0.8, 0.95, 1.0]
            ])
        ]
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_full_pipeline_from_matrices(self):
        """Test complete pipeline: matrices → ST graph → metrics."""
        # Step 1: Create ST graph from correlation matrices
        st_graph = spatio_temporal_graph_from_corr_matrices(
            self.matrices, self.areas, corr_thr=0.4
        )
        
        # Verify ST graph was created
        self.assertIsInstance(st_graph, SpatioTemporalGraph)
        self.assertGreater(len(st_graph.nodes), 0)
        self.assertGreater(len(st_graph.edges), 0)
        
        # Step 2: Calculate spatial metrics
        spatial_records = calculate_spatial_metrics(st_graph)
        self.assertEqual(len(spatial_records), 3)  # One per time point
        
        # Step 3: Calculate temporal metrics
        temporal_records = calculate_temporal_metrics(st_graph)
        self.assertEqual(len(temporal_records), 1)  # One for whole graph
        
        # Step 4: Convert to DataFrames
        spatial_df = pd.DataFrame(spatial_records)
        temporal_df = pd.DataFrame(temporal_records)
        
        # Verify DataFrames have expected structure
        self.assertIn('Time', spatial_df.columns)
        self.assertGreater(len(spatial_df.columns), 1)
        self.assertGreater(len(temporal_df.columns), 0)
    
    def test_round_trip_save_load(self):
        """Test save/load round trip preserves data for analysis."""
        # Create ST graph
        st_graph = spatio_temporal_graph_from_corr_matrices(
            self.matrices, self.areas, corr_thr=0.4
        )
        
        # Save to file
        save_path = Path(self.temp_dir) / 'test_roundtrip.zip'
        save_spatio_temporal_graph(st_graph, save_path)
        
        # Load back
        loaded_graph = load_spatio_temporal_graph(save_path)
        
        # Verify structure is preserved
        self.assertEqual(len(loaded_graph.nodes), len(st_graph.nodes))
        self.assertEqual(len(loaded_graph.edges), len(st_graph.edges))
        
        # Calculate metrics on both and compare
        original_spatial = calculate_spatial_metrics(st_graph)
        loaded_spatial = calculate_spatial_metrics(loaded_graph)
        
        original_temporal = calculate_temporal_metrics(st_graph)
        loaded_temporal = calculate_temporal_metrics(loaded_graph)
        
        # Metrics should be very similar (allowing for numerical precision)
        for orig, loaded in zip(original_spatial, loaded_spatial):
            for key in orig:
                if isinstance(orig[key], (int, float)) and not np.isnan(orig[key]):
                    self.assertAlmostEqual(orig[key], loaded[key], places=6)
        
        for key in original_temporal[0]:
            if isinstance(original_temporal[0][key], (int, float)):
                self.assertAlmostEqual(original_temporal[0][key], loaded_temporal[0][key], places=6)
    
    def test_metrics_save_load_roundtrip(self):
        """Test metrics save/load round trip."""
        # Create ST graph and calculate metrics
        st_graph = spatio_temporal_graph_from_corr_matrices(
            self.matrices, self.areas, corr_thr=0.4
        )
        
        spatial_metrics = pd.DataFrame(calculate_spatial_metrics(st_graph))
        spatial_metrics.set_index(pd.Index(['subject']*len(spatial_metrics), name='id'), inplace=True)
        temporal_metrics = pd.DataFrame(calculate_temporal_metrics(st_graph))
        temporal_metrics.set_index(pd.Index(['subject']*len(temporal_metrics), name='id'), inplace=True)
        
        # Save metrics
        spatial_path = Path(self.temp_dir) / 'spatial_metrics.csv'
        temporal_path = Path(self.temp_dir) / 'temporal_metrics.csv'
        
        from fstg_toolkit.io import save_metrics
        save_metrics(spatial_path, spatial_metrics)
        save_metrics(temporal_path, temporal_metrics)
        
        # Load metrics back
        loaded_spatial = load_metrics(spatial_path)
        loaded_temporal = load_metrics(temporal_path)
        
        # Verify data is preserved
        pd.testing.assert_frame_equal(loaded_spatial, spatial_metrics)
        pd.testing.assert_frame_equal(loaded_temporal, temporal_metrics)


class SimulationToAnalysisIntegrationTestCase(unittest.TestCase):
    """Test integration between simulation and analysis components."""
    
    def setUp(self):
        """Set up test patterns."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test patterns
        self.pattern1 = generate_pattern(
            networks_list=[
                [((1, 3), 1, 0.8), ((4, 6), 2, -0.7)],
                [((1, 4), 1, 0.9), ((5, 6), 2, -0.6)]
            ],
            spatial_edges=[(1, 2, 0.5)],
            temporal_edges=[(1, 3, 'pp'), (2, 4, 'po')]
        )
        
        self.pattern2 = generate_pattern(
            networks_list=[
                [((1, 4), 1, 0.9), ((5, 6), 2, -0.6)],
                [((1, 3), 1, 0.8), ((4, 6), 2, -0.7)]
            ],
            spatial_edges=[(1, 2, 0.6)],
            temporal_edges=[(1, (2, 3), 'split'), (2, 4, 'eq')]
        )
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_simulate_and_analyze(self):
        """Test simulating a graph and then analyzing it."""
        # Create simulator and generate graph
        simulator = SpatioTemporalGraphSimulator(p1=self.pattern1, p2=self.pattern2)
        simulated_graph = simulator.simulate('p1', 2, 'p2', 1, 'p1')
        
        # Verify simulated graph
        self.assertIsInstance(simulated_graph, SpatioTemporalGraph)
        self.assertGreater(len(simulated_graph.nodes), 0)
        self.assertGreater(len(simulated_graph.edges), 0)
        
        # Calculate metrics on simulated graph
        spatial_metrics = calculate_spatial_metrics(simulated_graph)
        temporal_metrics = calculate_temporal_metrics(simulated_graph)
        
        # Verify metrics were calculated
        self.assertGreater(len(spatial_metrics), 0)
        self.assertGreater(len(temporal_metrics), 0)
        
        # Convert to DataFrames
        spatial_df = pd.DataFrame(spatial_metrics)
        temporal_df = pd.DataFrame(temporal_metrics)
        
        # Verify DataFrames have expected content
        self.assertIn('Time', spatial_df.columns)
        self.assertGreaterEqual(len(spatial_df), simulated_graph.graph['max_time'] + 1)
    
    def test_simulation_reproducibility(self):
        """Test that simulation produces reproducible results."""
        # Create simulator
        simulator = SpatioTemporalGraphSimulator(p1=self.pattern1)
        
        # Generate same sequence twice
        graph1 = simulator.simulate('p1', 1, 'p1')
        graph2 = simulator.simulate('p1', 1, 'p1')
        
        # Results should be identical (deterministic)
        self.assertEqual(len(graph1.nodes), len(graph2.nodes))
        self.assertEqual(len(graph1.edges), len(graph2.edges))
        
        # Calculate same metrics
        metrics1 = calculate_temporal_metrics(graph1)
        metrics2 = calculate_temporal_metrics(graph2)
        
        # Metrics should be identical
        for key in metrics1[0]:
            self.assertEqual(metrics1[0][key], metrics2[0][key])


class RegressionTestCase(unittest.TestCase):
    """Test for regression - ensure known behaviors are preserved."""
    
    def test_known_graph_structure_preserved(self):
        """Test that known graph structures are preserved through operations."""
        # Create a known graph structure
        areas = pd.DataFrame({
            'Id_Area': [1, 2, 3, 4],
            'Name_Area': ['A1', 'A2', 'A3', 'A4'],
            'Name_Region': ['R1', 'R1', 'R2', 'R2']
        })
        areas.set_index('Id_Area', inplace=True)
        
        # Known correlation matrices
        matrices = [
            np.array([
                [1.0, 0.8, 0.1, 0.0],
                [0.8, 1.0, 0.2, 0.1],
                [0.1, 0.2, 1.0, 0.9],
                [0.0, 0.1, 0.9, 1.0]
            ]),
            np.array([
                [1.0, 0.9, 0.2, 0.1],
                [0.9, 1.0, 0.3, 0.2],
                [0.2, 0.3, 1.0, 0.8],
                [0.1, 0.2, 0.8, 1.0]
            ])
        ]
        
        # Create ST graph
        st_graph = spatio_temporal_graph_from_corr_matrices(matrices, areas, corr_thr=0.4)
        
        # Save and load
        temp_file = Path(tempfile.gettempdir()) / 'regression_test.zip'
        save_spatio_temporal_graph(st_graph, temp_file)
        loaded_graph = load_spatio_temporal_graph(temp_file)
        
        # Verify structure is preserved
        self.assertEqual(len(loaded_graph.nodes), len(st_graph.nodes))
        self.assertEqual(len(loaded_graph.edges), len(st_graph.edges))
        
        # Verify metrics are consistent
        orig_spatial = calculate_spatial_metrics(st_graph)
        loaded_spatial = calculate_spatial_metrics(loaded_graph)
        
        orig_temporal = calculate_temporal_metrics(st_graph)
        loaded_temporal = calculate_temporal_metrics(loaded_graph)
        
        # Spatial metrics should be identical
        for o, l in zip(orig_spatial, loaded_spatial):
            for key in o:
                if isinstance(o[key], (int, float)) and not np.isnan(o[key]):
                    self.assertAlmostEqual(o[key], l[key], places=6)
        
        # Temporal metrics should be identical
        for key in orig_temporal[0]:
            if isinstance(orig_temporal[0][key], (int, float)):
                self.assertAlmostEqual(orig_temporal[0][key], loaded_temporal[0][key], places=6)
        
        # Clean up
        temp_file.unlink()
    
    def test_backward_compatibility_data_formats(self):
        """Test backward compatibility with data formats."""
        # This would test loading older format data files
        # For now, just test that current format can be loaded
        
        # Create a graph and save it
        areas = pd.DataFrame({
            'Id_Area': [1, 2],
            'Name_Area': ['A1', 'A2'],
            'Name_Region': ['R1', 'R1']
        })
        areas.set_index('Id_Area', inplace=True)
        
        graph = nx.DiGraph()
        graph.add_nodes_from([
            (1, dict(t=0, areas={1}, region='R1', internal_strength=1.0)),
            (2, dict(t=1, areas={1, 2}, region='R1', internal_strength=0.8))
        ])
        graph.add_edge(1, 2, type='temporal', transition='PP')
        graph.graph['min_time'] = 0
        graph.graph['max_time'] = 1
        
        st_graph = SpatioTemporalGraph(graph, areas)
        
        # Save and load
        temp_file = Path(tempfile.gettempdir()) / 'compat_test.zip'
        save_spatio_temporal_graph(st_graph, temp_file)
        loaded_graph = load_spatio_temporal_graph(temp_file)
        
        # Should load without errors and have correct structure
        self.assertEqual(len(loaded_graph.nodes), 2)
        self.assertEqual(len(loaded_graph.edges), 1)
        
        # Clean up
        temp_file.unlink()
    
    def test_numerical_stability(self):
        """Test numerical stability of operations."""
        # Create matrices with values very close to threshold
        areas = pd.DataFrame({
            'Id': [1, 2, 3],
            'Name_Area': ['A1', 'A2', 'A3'],
            'Name_Region': ['R1', 'R1', 'R1']
        })
        areas.set_index('Id', inplace=True)
        
        # Matrices with correlations very close to threshold (0.4)
        matrices = [
            np.array([
                [1.0, 0.4001, 0.3999],
                [0.4001, 1.0, 0.4000],
                [0.3999, 0.4000, 1.0]
            ]),
            np.array([
                [1.0, 0.3999, 0.4001],
                [0.3999, 1.0, 0.4000],
                [0.4001, 0.4000, 1.0]
            ])
        ]
        
        # This should not crash or produce inconsistent results
        st_graph = spatio_temporal_graph_from_corr_matrices(matrices, areas, corr_thr=0.4)
        
        # Should produce consistent results
        spatial_metrics = calculate_spatial_metrics(st_graph)
        temporal_metrics = calculate_temporal_metrics(st_graph)
        
        # Metrics should be finite
        for record in spatial_metrics:
            for value in record.values():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    self.assertTrue(np.isfinite(value))
        
        for value in temporal_metrics[0].values():
            if isinstance(value, (int, float)):
                self.assertTrue(np.isfinite(value))


class PerformanceRegressionTestCase(unittest.TestCase):
    """Test that performance doesn't degrade significantly."""
    
    def test_large_graph_creation_performance(self):
        """Test that creating large graphs doesn't take excessive time."""
        import time
        
        # Create areas for a larger graph
        num_areas = 50
        areas = pd.DataFrame({
            'Id': range(1, num_areas + 1),
            'Name_Area': [f'A{i}' for i in range(1, num_areas + 1)],
            'Name_Region': ['R1'] * (num_areas // 2) + ['R2'] * (num_areas - num_areas // 2)
        })
        areas.set_index('Id', inplace=True)
        
        # Create correlation matrices (this should be reasonably fast)
        start_time = time.time()
        
        # Create a simple pattern with many areas
        pattern = generate_pattern(
            networks_list=[
                [((1, num_areas), 1, 0.8)]
            ],
            spatial_edges=[],
            temporal_edges=[]
        )
        
        # Simulate correlation matrices
        simulator = CorrelationMatrixSequenceSimulator(
            pattern, threshold=0.4, rng=np.random.default_rng(42)
        )
        matrices = simulator.simulate()
        
        end_time = time.time()
        
        # Should complete in reasonable time (this is a regression test)
        # If this starts taking much longer, it indicates a performance issue
        self.assertLess(end_time - start_time, 10.0)  # Should take < 10 seconds
        
        # Verify the result
        self.assertEqual(matrices.shape, (1, num_areas, num_areas))
    
    def test_metrics_calculation_performance(self):
        """Test that metrics calculation on large graphs is reasonable."""
        import time
        
        # Create a reasonably sized graph
        areas = pd.DataFrame({
            'Id': range(1, 21),  # 20 areas
            'Name_Area': [f'A{i}' for i in range(1, 21)],
            'Name_Region': ['R1'] * 10 + ['R2'] * 10
        })
        areas.set_index('Id', inplace=True)
        
        # Create correlation matrices for 5 time points
        matrices = []
        for i in range(5):
            matrix = np.eye(20)
            # Fill with some correlations
            for j in range(20):
                for k in range(j + 1, 20):
                    if j < 10 and k < 10:  # Within R1
                        matrix[j, k] = matrix[k, j] = 0.8 - (j + k) * 0.01
                    elif j >= 10 and k >= 10:  # Within R2
                        matrix[j, k] = matrix[k, j] = 0.7 - (j + k - 20) * 0.01
                    else:  # Between regions
                        matrix[j, k] = matrix[k, j] = 0.3 + (j + k) * 0.005
            matrices.append(matrix)
        
        # Create ST graph
        start_time = time.time()
        st_graph = spatio_temporal_graph_from_corr_matrices(matrices, areas, corr_thr=0.2)
        
        # Calculate metrics
        spatial_metrics = calculate_spatial_metrics(st_graph)
        temporal_metrics = calculate_temporal_metrics(st_graph)
        end_time = time.time()
        
        # Should complete in reasonable time
        self.assertLess(end_time - start_time, 5.0)  # Should take < 5 seconds
        
        # Verify results
        self.assertEqual(len(spatial_metrics), 5)
        self.assertEqual(len(temporal_metrics), 1)
