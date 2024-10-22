import unittest

from test_simulation import CorrelationMatrixSimulationTestCase


def simulation_suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(CorrelationMatrixSimulationTestCase))
    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(simulation_suite())
