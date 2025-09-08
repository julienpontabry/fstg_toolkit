import unittest

from fmri_st_graph.measures import calculate_spatial_measures, calculate_temporal_measures
from fmri_st_graph.measures import get_spatial_measures_registry, get_temporal_measures_registry
from fmri_st_graph.io import load_spatio_temporal_graph
from test_common import graph_path


class MeasuresCalculationTestCase(unittest.TestCase):
    def setUp(self):
        self.graph = load_spatio_temporal_graph(graph_path)

    def test_calculate_spatial_measures(self):
        df = calculate_spatial_measures(self.graph)
        
        registry = list(get_spatial_measures_registry())
        for measure, _ in registry:
            self.assertIn(measure, df.columns)

        self.assertEqual(len(df), len(self.graph.time_range))

    def test_calculate_temporal_measures(self):
        df = calculate_temporal_measures(self.graph)

        registry = list(get_temporal_measures_registry())
        for measure, _ in registry:
            self.assertIn(measure, df.columns)

        self.assertEqual(len(df), 1)


# TODO test also the measured values
