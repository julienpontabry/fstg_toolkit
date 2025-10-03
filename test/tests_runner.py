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
