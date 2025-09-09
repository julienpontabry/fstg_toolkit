import unittest

from test_factory import SpatioTemporalGraphFactoryTestCase
from test_simulation import CorrelationMatrixSimulationTestCase, SpatioTemporalGraphSimulationTestCase
from test_metrics import MetricsCalculationTestCase


def simulation_suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(CorrelationMatrixSimulationTestCase))
    suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(SpatioTemporalGraphSimulationTestCase))
    return suite

def factory_suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(SpatioTemporalGraphFactoryTestCase))
    return suite

def measures_suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(MetricsCalculationTestCase))
    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()

    total_failed = 0

    for suite in (simulation_suite(), factory_suite(), measures_suite()):
        result = runner.run(suite)
        total_failed += len(result.failures)

    if total_failed > 0:
        exit(1)
