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


import io
import unittest
import zipfile
from pathlib import Path

import matplotlib
import numpy as np
import numpy.testing as npt
import pandas as pd

# A non-interactive backend is mandatory before pyplot is imported.
matplotlib.use('Agg')

import matplotlib.pyplot as plt

from fstg_toolkit import visualization
from fstg_toolkit.factory import spatio_temporal_graph_from_corr_matrices
from fstg_toolkit.graph import RC5

DATA_DIR = Path(__file__).parent / 'data'


def _private(name: str):
    """Fetch a module-level private helper of the visualization module.

    A plain `visualization.__helper` reference written inside a class body is
    rewritten by Python's name mangling into `_ClassName__helper`, so the
    lookup has to go through `getattr`.

    Parameters
    ----------
    name: str
        The name of the helper, including its leading double underscore.

    Returns
    -------
    Any
        The requested helper.
    """
    return getattr(visualization, name)


def _load_toy_graph():
    """Build the toy spatio-temporal graph shared by the test cases.

    Returns
    -------
    SpatioTemporalGraph
        A graph of 8 nodes spanning 3 time points.
    """
    with zipfile.ZipFile(DATA_DIR / 'toy-example_graph.zip') as archive:
        areas = pd.read_csv(io.BytesIO(archive.read('areas.csv')), index_col='Id_Area')

    # The fixture stores one 2D matrix per time point, whereas the factory
    # expects the whole sequence as a single stacked array.
    with np.load(DATA_DIR / 'toy-example_matrices.npz') as raw:
        matrices = np.stack([raw[name] for name in raw])

    return spatio_temporal_graph_from_corr_matrices(matrices, areas)


class GeometryHelpersTestCase(unittest.TestCase):
    """Test the pure geometry and color helpers backing the plots."""

    def test_polar2cart_on_the_x_axis(self):
        """A null angle maps to the positive x-axis."""
        x, y = _private('__polar2cart')(np.array([0.0]), 2.0)
        npt.assert_allclose(x, [2.0])
        npt.assert_allclose(y, [0.0], atol=1e-12)

    def test_polar2cart_on_the_y_axis(self):
        """A quarter turn maps to the positive y-axis."""
        x, y = _private('__polar2cart')(np.array([np.pi / 2]), 2.0)
        npt.assert_allclose(x, [0.0], atol=1e-12)
        npt.assert_allclose(y, [2.0])

    def test_polar2cart_is_vectorised(self):
        """Several angles are converted in one call."""
        x, y = _private('__polar2cart')(np.array([0.0, np.pi]), 1.0)
        self.assertEqual(x.shape, (2,))
        npt.assert_allclose(x, [1.0, -1.0], atol=1e-12)

    def test_angle_between_along_axes(self):
        """The angle is measured counterclockwise, in degrees, within [0, 360)."""
        angle_between = _private('__angle_between')
        self.assertAlmostEqual(angle_between((0, 0), (1, 0)), 0.0)
        self.assertAlmostEqual(angle_between((0, 0), (0, 1)), 90.0)
        self.assertAlmostEqual(angle_between((0, 0), (-1, 0)), 180.0)

    def test_angle_between_is_never_negative(self):
        """A clockwise direction is reported as its positive equivalent."""
        self.assertAlmostEqual(_private('__angle_between')((0, 0), (0, -1)), 270.0)

    def test_readable_annotation_keeps_right_hand_side_upright(self):
        """An angle on the right half is used as is, anchored on its left."""
        self.assertEqual(_private('__readable_angled_annotation')(45),
                         {'rotation': 45, 'ha': 'left'})

    def test_readable_annotation_flips_left_hand_side(self):
        """An angle on the left half is flipped so the text stays readable."""
        self.assertEqual(_private('__readable_angled_annotation')(180),
                         {'rotation': 360, 'ha': 'right'})

    def test_readable_annotation_at_the_boundaries(self):
        """Both boundaries of the right half are anchored on the left."""
        readable = _private('__readable_angled_annotation')
        self.assertEqual(readable(90)['ha'], 'left')
        self.assertEqual(readable(270)['ha'], 'left')
        self.assertEqual(readable(271)['ha'], 'left')

    def test_edge_connection_style_is_straight_for_identical_angles(self):
        """Two nodes at the same angle are joined by a straight arc."""
        self.assertEqual(_private('__edge_con_style')(0.0, 0.0), 'arc3, rad=0.0')

    def test_edge_connection_style_bends_towards_close_nodes(self):
        """The closer the angles, the stronger the bending."""
        con_style = _private('__edge_con_style')
        close = float(con_style(0.0, 0.1).split('rad=')[1])
        far = float(con_style(0.0, np.pi / 2).split('rad=')[1])
        self.assertGreater(abs(close), abs(far))

    def test_edge_connection_style_takes_the_shortest_way_around(self):
        """Angles on either side of the origin are treated as neighbors."""
        con_style = _private('__edge_con_style')
        # 0.1 and 6.2 rad are barely a fifth of a radian apart going through
        # the origin, so the arc must bend as strongly as for close angles.
        wrapped = float(con_style(0.1, 6.2).split('rad=')[1])
        self.assertGreater(abs(wrapped), 0.5)

    def test_annotation_connection_style_reports_both_angles(self):
        """The annotation style carries the two angles it links."""
        style = _private('__annot_con_style')(0.0, np.pi / 2)
        self.assertTrue(style.startswith('angle,'))
        self.assertIn('angleA=0.0', style)

    def test_transition_colours_are_defined_for_every_relation(self):
        """Every RC5 relation maps to a color name."""
        trans_color = getattr(visualization, '_trans_color')
        for relation in RC5:
            with self.subTest(relation=relation.name):
                self.assertIsInstance(trans_color(relation), str)

    def test_transition_colours_distinguish_inclusions(self):
        """A proper part and its inverse are drawn with different colors."""
        trans_color = getattr(visualization, '_trans_color')
        self.assertNotEqual(trans_color(RC5.PP), trans_color(RC5.PPi))

    def test_inch2cm_divides_by_the_inch_length(self):
        """The conversion factor is the length of an inch in centimetres."""
        self.assertAlmostEqual(getattr(visualization, '_inch2cm')(2.54), 1.0)


class GraphLayoutTestCase(unittest.TestCase):
    """Test the helpers turning a graph into plotting coordinates."""

    @classmethod
    def setUpClass(cls):
        """Build the toy graph once for the whole test case."""
        cls.graph = _load_toy_graph()

    def test_multipartite_layout_places_every_node(self):
        """The layout yields one position per node."""
        layout = _private('__time_multipartite_layout')(self.graph)
        self.assertEqual(set(layout), set(self.graph.nodes))

    def test_multipartite_layout_uses_time_as_abscissa(self):
        """The x coordinate of a node is its time point."""
        layout = _private('__time_multipartite_layout')(self.graph)
        for node, (x, _) in layout.items():
            with self.subTest(node=node):
                self.assertEqual(x, self.graph.nodes[node]['t'])

    def test_multipartite_layout_separates_simultaneous_nodes(self):
        """Nodes sharing a time point get distinct ordinates."""
        layout = _private('__time_multipartite_layout')(self.graph)
        ordinates = [y for node, (x, y) in layout.items() if x == 0]
        self.assertEqual(len(ordinates), len(set(ordinates)))

    def test_areas_positions_covers_every_area(self):
        """One angular position is produced per area of the description."""
        areas, angles, x, y = _private('__areas_positions')(self.graph)
        self.assertEqual(len(angles), len(areas))
        self.assertEqual(x.shape, angles.shape)
        self.assertEqual(y.shape, angles.shape)

    def test_areas_positions_lie_on_a_circle(self):
        """Areas are laid out at a constant distance from the center."""
        _, _, x, y = _private('__areas_positions')(self.graph)
        radii = np.hypot(x, y)
        npt.assert_allclose(radii, np.full_like(radii, radii[0]), atol=1e-9)


class PlotTestCase(unittest.TestCase):
    """Test that the public plotting functions render onto a given axes."""

    @classmethod
    def setUpClass(cls):
        """Build the toy graph once for the whole test case."""
        cls.graph = _load_toy_graph()

    def setUp(self):
        """Set up a fresh figure for each test."""
        self.figure, self.axes = plt.subplots()

    def tearDown(self):
        """Close every figure opened by the test."""
        plt.close('all')

    def test_multipartite_plot_draws_nodes_and_labels(self):
        """The multipartite plot adds a node collection and area labels."""
        visualization.multipartite_plot(self.graph, ax=self.axes)
        self.assertGreater(len(self.axes.collections), 0)
        self.assertGreater(len(self.axes.texts), 0)

    def test_multipartite_plot_creates_its_own_axes(self):
        """Omitting the axes lets the function open a figure by itself."""
        plt.close('all')
        visualization.multipartite_plot(self.graph)
        self.assertEqual(len(plt.get_fignums()), 1)

    def test_spatial_plot_draws_the_first_time_point(self):
        """The spatial plot renders the connectivity of a single time point."""
        visualization.spatial_plot(self.graph, 0, ax=self.axes)
        self.assertGreater(len(self.axes.texts), 0)

    def test_spatial_plot_accepts_every_time_point(self):
        """Every time point of the graph can be rendered."""
        for time in range(self.graph.graph['max_time'] + 1):
            with self.subTest(time=time):
                _, axes = plt.subplots()
                visualization.spatial_plot(self.graph, time, ax=axes)
                self.assertGreater(len(axes.texts), 0)

    def test_temporal_plot_returns_coordinates_and_edges(self):
        """The temporal plot reports the coordinates it used and the edges it drew."""
        coordinates, edges = visualization.temporal_plot(self.graph, ax=self.axes)
        self.assertEqual(set(coordinates), set(self.graph.nodes))
        self.assertIsInstance(edges, dict)

    def test_temporal_plot_needs_the_efficiency_attribute(self):
        """A graph whose nodes lack the efficiency metric cannot be drawn."""
        # A fresh graph is loaded rather than copied, because copying a
        # SpatioTemporalGraph does not carry its areas description over.
        graph = _load_toy_graph()
        for _, data in graph.nodes(data=True):
            del data['efficiency']

        with self.assertRaises(KeyError):
            visualization.temporal_plot(graph, ax=self.axes)
