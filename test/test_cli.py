# Copyright 2025 ICube (University of Strasbourg - CNRS)
# author: Julien PONTABRY (ICube)
#
# This software is a computer program whose purpose is to provide a toolkit
# to model, process and analyze the longitudinal reorganization of brain
# connectivity data, as functional MRI for instance.
#
# This software is governed by the CeCILL license under French law and
# abiding by the rules of distribution of free software. You can use,
# modify and/or redistribute the software under the terms of the CeCILL
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
# knowledge of the CeCILL license and that you accept its terms.


import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import matplotlib
import numpy as np
from click import exceptions as click_exceptions
from click.testing import CliRunner

# A non-interactive backend is mandatory before pyplot is imported anywhere,
# because importing the CLI pulls in matplotlib.pyplot at module level.
matplotlib.use('Agg')

from fstg_toolkit.__main__ import cli, GRAPH_SEQUENCE_DESCRIPTION, NETWORKS_DESCRIPTION, \
    SPATIAL_EDGES_DESCRIPTION, TEMPORAL_EDGES_DESCRIPTION

DATA_DIR = Path(__file__).parent / 'data'


class _FakeMonitor:
    """Stand-in for a screeninfo monitor, so that plot commands run headless."""
    width: int = 1920
    height: int = 1080
    width_mm: float = 509
    height_mm: float = 286


def _patch_monitors() -> mock._patch:
    """Patch the monitor lookup used to size figures.

    The `plot` commands call `screeninfo.get_monitors()`, which raises on a
    headless machine such as a CI runner.

    Returns
    -------
    unittest.mock._patch
        The patcher to use as a context manager or decorator.
    """
    return mock.patch('fstg_toolkit.__main__.get_monitors', return_value=[_FakeMonitor()])


class ParamTypesTestCase(unittest.TestCase):
    """Test the custom click parameter types used to parse simulation descriptions."""

    def test_networks_single(self):
        """A single network description yields one group with one network."""
        self.assertEqual(NETWORKS_DESCRIPTION.convert('0:3,0,0.8', None, None),
                         [[((0, 3), 0, 0.8)]])

    def test_networks_several_groups(self):
        """Networks separated by a slash are parsed as independent groups."""
        self.assertEqual(NETWORKS_DESCRIPTION.convert('0:3,0,0.8/4:6,1,0.5', None, None),
                         [[((0, 3), 0, 0.8)], [((4, 6), 1, 0.5)]])

    def test_networks_single_area_and_negative_strength(self):
        """An area range may be a lone index and the strength may be negative."""
        self.assertEqual(NETWORKS_DESCRIPTION.convert('5,2,-0.3', None, None),
                         [[(5, 2, -0.3)]])

    def test_networks_rejects_non_string(self):
        """A non-string value is rejected rather than silently coerced."""
        with self.assertRaises(click_exceptions.BadParameter):
            NETWORKS_DESCRIPTION.convert(42, None, None)

    def test_spatial_edges_single(self):
        """A spatial edge is parsed as a (node, node, correlation) triple."""
        self.assertEqual(SPATIAL_EDGES_DESCRIPTION.convert('0,1,0.5', None, None),
                         [(0, 1, 0.5)])

    def test_spatial_edges_several(self):
        """Several space-separated spatial edges are all parsed."""
        self.assertEqual(SPATIAL_EDGES_DESCRIPTION.convert('0,1,0.5 1,2,-0.7', None, None),
                         [(0, 1, 0.5), (1, 2, -0.7)])

    def test_temporal_edge_equality(self):
        """Two single nodes describe an equality transition."""
        self.assertEqual(TEMPORAL_EDGES_DESCRIPTION.convert('0,1', None, None),
                         [(0, 1, 'eq')])

    def test_temporal_edge_split(self):
        """A single node towards a range describes a split transition."""
        self.assertEqual(TEMPORAL_EDGES_DESCRIPTION.convert('0,1-2', None, None),
                         [(0, (1, 2), 'split')])

    def test_temporal_edge_merge(self):
        """A range towards a single node describes a merge transition."""
        self.assertEqual(TEMPORAL_EDGES_DESCRIPTION.convert('0-1,2', None, None),
                         [((0, 1), 2, 'merge')])

    def test_temporal_edge_range_to_range_is_rejected(self):
        """A transition between two ranges is not supported."""
        with self.assertRaises(click_exceptions.BadParameter):
            TEMPORAL_EDGES_DESCRIPTION.convert('0-1,2-3', None, None)

    def test_sequence_mixes_patterns_and_spacings(self):
        """A sequence alternates pattern names and steady state counts."""
        self.assertEqual(GRAPH_SEQUENCE_DESCRIPTION.convert('p1 3 p2', None, None),
                         ['p1', 3, 'p2'])

    def test_sequence_is_case_insensitive(self):
        """Pattern names are normalized to lower case."""
        self.assertEqual(GRAPH_SEQUENCE_DESCRIPTION.convert('P1 2 P2', None, None),
                         ['p1', 2, 'p2'])

    def test_sequence_rejects_unknown_token(self):
        """A token that is neither a pattern nor a number is rejected."""
        with self.assertRaises(click_exceptions.BadParameter):
            GRAPH_SEQUENCE_DESCRIPTION.convert('xx', None, None)


class CliTestCase(unittest.TestCase):
    """Test the command line interface end to end on a small built dataset."""

    @classmethod
    def setUpClass(cls):
        """Build, once, the inputs shared by every command test.

        The committed `toy-example_graph.zip` predates the `efficiency` node
        attribute and cannot be plotted, so a fresh archive is built here from
        the raw matrices instead.
        """
        cls.temp_dir = Path(tempfile.mkdtemp())
        cls.areas_path = cls.temp_dir / 'areas.csv'
        cls.matrices_path = cls.temp_dir / 'matrices.npz'
        cls.graph_path = cls.temp_dir / 'built.zip'

        with zipfile.ZipFile(DATA_DIR / 'toy-example_graph.zip') as archive:
            cls.areas_path.write_bytes(archive.read('areas.csv'))

        # The factory expects one sequence of matrices per subject, so the
        # separate 2D arrays of the fixture are stacked along the time axis.
        with np.load(DATA_DIR / 'toy-example_matrices.npz') as raw:
            np.savez(cls.matrices_path, subject1=np.stack([raw[name] for name in raw]))

        result = CliRunner().invoke(cli, ['graph', 'build', '-o', str(cls.graph_path),
                                          str(cls.areas_path), str(cls.matrices_path)])
        assert result.exit_code == 0, result.output

    @classmethod
    def tearDownClass(cls):
        """Remove the shared temporary directory."""
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def setUp(self):
        """Set up a runner and a private copy of the built archive."""
        self.runner = CliRunner()
        self.work_dir = Path(tempfile.mkdtemp())
        # Several commands write back into the archive they are given, so each
        # test works on its own copy.
        self.graph_copy = self.work_dir / 'graph.zip'
        shutil.copy(self.graph_path, self.graph_copy)

    def tearDown(self):
        """Remove the per-test temporary directory."""
        shutil.rmtree(self.work_dir, ignore_errors=True)

    def test_help_of_every_command(self):
        """Every command exposes a help page, which proves the click tree is sound."""
        for args in ([], ['graph'], ['graph', 'build'], ['graph', 'metrics'],
                     ['graph', 'frequent'], ['graph', 'simulate'],
                     ['graph', 'simulate', 'pattern'], ['graph', 'simulate', 'sequence'],
                     ['graph', 'simulate', 'correlations'], ['dashboard'],
                     ['dashboard', 'show'], ['dashboard', 'serve']):
            with self.subTest(command=' '.join(args) or '<root>'):
                result = self.runner.invoke(cli, args + ['--help'])
                self.assertEqual(result.exit_code, 0, result.output)

    def test_plot_help_of_every_subcommand(self):
        """The plot group takes the archive before the subcommand."""
        for sub in ('multipartite', 'spatial', 'temporal', 'dynamic'):
            with self.subTest(subcommand=sub):
                result = self.runner.invoke(cli, ['plot', str(self.graph_copy), sub, '--help'])
                self.assertEqual(result.exit_code, 0, result.output)

    def test_build_produces_a_readable_archive(self):
        """Building writes an archive holding the graph, the areas and the raw matrices."""
        output = self.work_dir / 'out.zip'
        result = self.runner.invoke(cli, ['graph', 'build', '-o', str(output),
                                          str(self.areas_path), str(self.matrices_path)])
        self.assertEqual(result.exit_code, 0, result.output)
        with zipfile.ZipFile(output) as archive:
            self.assertIn('areas.csv', archive.namelist())
            self.assertIn('subject1.json', archive.namelist())
            self.assertIn('subject1.npy', archive.namelist())

    def test_build_without_raw_matrices(self):
        """The --no-raw flag omits the correlation matrices from the archive."""
        output = self.work_dir / 'out.zip'
        result = self.runner.invoke(cli, ['graph', 'build', '--no-raw', '-o', str(output),
                                          str(self.areas_path), str(self.matrices_path)])
        self.assertEqual(result.exit_code, 0, result.output)
        with zipfile.ZipFile(output) as archive:
            self.assertNotIn('subject1.npy', archive.namelist())

    def test_build_rejects_a_missing_areas_file(self):
        """A missing input path is caught by click before the command runs."""
        result = self.runner.invoke(cli, ['graph', 'build', '-o', str(self.work_dir / 'o.zip'),
                                          str(self.work_dir / 'nope.csv'), str(self.matrices_path)])
        self.assertEqual(result.exit_code, 2)

    def test_metrics_adds_metrics_to_the_archive(self):
        """Computing metrics writes them back into the archive."""
        result = self.runner.invoke(cli, ['graph', 'metrics', str(self.graph_copy)])
        self.assertEqual(result.exit_code, 0, result.output)
        with zipfile.ZipFile(self.graph_copy) as archive:
            self.assertTrue(any(name.endswith('.csv') and name != 'areas.csv'
                                for name in archive.namelist()),
                            f"no metrics file in {archive.namelist()}")

    def test_plot_commands_run_headless(self):
        """Each static plot command renders without raising."""
        for sub in ('multipartite', 'spatial', 'temporal'):
            with self.subTest(subcommand=sub), _patch_monitors():
                result = self.runner.invoke(cli, ['plot', str(self.graph_copy), sub])
                self.assertEqual(result.exit_code, 0, result.output)

    def test_plot_spatial_accepts_a_time_index(self):
        """The spatial plot can be asked for a given time point."""
        with _patch_monitors():
            result = self.runner.invoke(cli, ['plot', str(self.graph_copy), 'spatial', '-t', '1'])
        self.assertEqual(result.exit_code, 0, result.output)

    def test_plot_spatial_reports_an_out_of_range_time(self):
        """A time index beyond the last one is reported instead of crashing."""
        with _patch_monitors():
            result = self.runner.invoke(cli, ['plot', str(self.graph_copy), 'spatial',
                                              '-t', '999'])
        self.assertEqual(result.exit_code, 0, result.output)

    def test_simulate_pattern_writes_a_loadable_pattern(self):
        """A pattern description is turned into an archive holding one graph."""
        output = self.work_dir / 'pattern.zip'
        result = self.runner.invoke(cli, ['graph', 'simulate', '-o', str(output), 'pattern',
                                          '0:3,0,0.8/4:6,1,0.5', '0,1,0.5'])
        self.assertEqual(result.exit_code, 0, result.output)
        with zipfile.ZipFile(output) as archive:
            self.assertIn('areas.csv', archive.namelist())
            self.assertTrue(any(name.endswith('.json') for name in archive.namelist()))
