import unittest

import pandas as pd

from fmri_st_graph.metrics import calculate_spatial_metrics, calculate_temporal_metrics
from fmri_st_graph.metrics import get_spatial_metrics_registry, get_temporal_metrics_registry
from fmri_st_graph.io import load_spatio_temporal_graph
from test_common import graph_path


class MetricsCalculationTestCase(unittest.TestCase):
    def setUp(self):
        self.graph = load_spatio_temporal_graph(graph_path)

    def test_calculate_metrics(self):
        df = pd.DataFrame.from_records(calculate_spatial_metrics(self.graph))
        
        registry = list(get_spatial_metrics_registry())
        for metric, _ in registry:
            self.assertIn(metric, df.columns)

        self.assertEqual(len(df), len(self.graph.time_range))

    def test_calculate_temporal_metrics(self):
        df = pd.DataFrame.from_records(calculate_temporal_metrics(self.graph))

        registry = list(get_temporal_metrics_registry())
        for metric, _ in registry:
            self.assertIn(metric, df.columns)

        self.assertEqual(len(df), 1)


# TODO test also the metrics
