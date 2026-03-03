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

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import cached_property
from math import cos, sin

import numpy as np
from numpy import pi


@dataclass(frozen=True)
class Arc:
    """An arc on the unit circle defined by start and end angles (in radians).

    Parameters
    ----------
    begin: float
        Start angle in radians.
    end: float
        End angle in radians.
    """

    begin: float
    end: float

    @cached_property
    def angle(self) -> float:
        """Angular span of the arc in radians (``end - begin``)."""
        return self.end - self.begin

    def sample(self, radius: float, angular_res: float = pi/64) -> np.ndarray:
        """Sample points along the arc at the given radius.

        Parameters
        ----------
        radius: float
            Radial distance from the origin.
        angular_res: float, optional
            Angular resolution between samples (default ``π/64``).

        Returns
        -------
        numpy.ndarray
            Complex-valued array of sampled points (real=x, imag=y).
        """
        n_samples = max(int(round(self.angle / angular_res)), 2)
        theta = np.linspace(self.begin, self.end, n_samples)
        return radius * np.exp(1j * theta)

    @staticmethod
    def from_proportions(proportions: list[float], begin: float = 0, length: float = 2*pi,
                         gap_size: float = 0.005) -> list['Arc']:
        """Create a list of arcs from proportional sizes with uniform gaps.

        Parameters
        ----------
        proportions: list[float]
            Relative sizes of each arc. Need not sum to 1.
        begin: float, optional
            Starting angle in radians (default 0).
        length: float, optional
            Total angular length to distribute (default ``2π``).
        gap_size: float, optional
            Gap between arcs as a fraction of the full circle (default 0.005).

        Returns
        -------
        list[Arc]
            One :class:`Arc` per proportion, laid out consecutively with gaps.
        """
        gap = 2*pi * gap_size

        if length < 2*pi:
            length += gap

        arcs = []

        for prop in proportions:
            end = begin + prop * length - gap
            arcs.append(Arc(begin, end))
            begin = end + gap

        return arcs


@dataclass(frozen=True)
class Ribbon:
    """A curved ribbon connecting two angular positions on the unit circle via a quadratic Bezier.

    Parameters
    ----------
    begin: float
        Angle (radians) of the ribbon's start point on the circle.
    end: float
        Angle (radians) of the ribbon's end point on the circle.
    """

    begin: float
    end: float

    @staticmethod
    def __pol2cart(theta: float) -> np.complex128:
        """Convert a polar angle to a unit-circle Cartesian complex number."""
        return np.exp(1j * theta)

    @cached_property
    def __begin_cart(self) -> np.complex128:
        """Cartesian position of the start point on the unit circle."""
        return self.__pol2cart(self.begin)

    @cached_property
    def __end_cart(self) -> np.complex128:
        """Cartesian position of the end point on the unit circle."""
        return self.__pol2cart(self.end)

    @cached_property
    def __middle(self) -> np.complex128:
        """Midpoint between the start and end Cartesian positions (used as Bezier control)."""
        return (self.__begin_cart + self.__end_cart) / 2

    @staticmethod
    def __bezier_quad(a: np.complex128, b: np.complex128, c: np.complex128, t: np.ndarray) -> np.ndarray:
        """Evaluate a quadratic Bezier curve.

        Parameters
        ----------
        a: complex
            Start point.
        b: complex
            End point.
        c: complex
            Control point.
        t: numpy.ndarray
            Parameter values in [0, 1].

        Returns
        -------
        numpy.ndarray
            Array of shape ``(2, len(t))`` with x- and y-coordinates.
        """
        x = (1-t)**2 * a.real + 2*(1-t)*t * c.real + t**2 * b.real
        y = (1 - t) ** 2 * a.imag + 2 * (1 - t) * t * c.imag + t**2 * b.imag
        return np.array([x, y])

    def sample(self, radius: float = 1.0, origin: tuple[float, float] = (0, 0),
               strength: float = 0.3, nb_points: int = 50) -> np.ndarray:
        """Sample points along the ribbon curve.

        Parameters
        ----------
        radius: float, optional
            Circle radius at which the ribbon endpoints sit (default 1.0).
        origin: tuple[float, float], optional
            Centre of the circle (default ``(0, 0)``).
        strength: float, optional
            Pull strength of the Bezier control point towards the origin
            (default 0.3).
        nb_points: int, optional
            Number of sample points along the curve (default 50).

        Returns
        -------
        numpy.ndarray
            Array of shape ``(2, nb_points)`` with x- and y-coordinates.
        """
        origin = origin[0] + 1j*origin[1]
        return self.__bezier_quad(
            self.__begin_cart * radius + origin,
            self.__end_cart * radius + origin,
            self.__middle * strength + origin,
            np.linspace(0, 1, nb_points))


@dataclass(frozen=True)
class Line:
    """A straight line segment defined by a length, orientation, and origin.

    Parameters
    ----------
    length: float
        Length of the segment.
    orientation: float
        Angle of the segment in radians (measured from the positive x-axis).
    origin: tuple[float, float], optional
        Start point of the segment (default ``(0, 0)``).
    """

    length: float
    orientation: float
    origin: tuple[float, float] = (0, 0)

    @cached_property
    def terminus(self) -> tuple[float, float]:
        """End point of the line segment computed from length and orientation."""
        x = self.length * cos(self.orientation) + self.origin[0]
        y = self.length * sin(self.orientation) + self.origin[1]
        return x, y

    def sample(self, offset: float = 0) -> np.ndarray:
        """Sample the two endpoints of the line, optionally perpendicular-shifted.

        Parameters
        ----------
        offset: float, optional
            Perpendicular offset from the centreline (default 0).

        Returns
        -------
        numpy.ndarray
            Array of shape ``(2, 2)`` containing the start and end points.
        """
        cos_theta = cos(self.orientation)
        sin_theta = sin(self.orientation)

        ox = -offset * sin_theta + self.origin[0]
        oy = offset * cos_theta + self.origin[1]
        x = self.length * cos_theta - offset * sin_theta + self.origin[0]
        y = self.length * sin_theta + offset * cos_theta + self.origin[1]

        return np.array([(ox, oy), (x, y)])

    @staticmethod
    def from_proportions(proportions: list[float], total_length: float, orientation: float,
                         origin: tuple[float, float] = (0, 0), gap_length: float = 0.25) -> list['Line']:
        """Create proportionally-sized line segments laid end-to-end with gaps.

        Parameters
        ----------
        proportions: list[float]
            Relative sizes of each segment.
        total_length: float
            Total available length (gaps included).
        orientation: float
            Shared orientation angle in radians.
        origin: tuple[float, float], optional
            Starting point for the first segment (default ``(0, 0)``).
        gap_length: float, optional
            Fixed gap length between consecutive segments (default 0.25).

        Returns
        -------
        list[Line]
            One :class:`Line` per proportion, laid out consecutively with gaps.
        """
        lines = []
        usable_length = total_length - gap_length * (len(proportions) - 1)
        terminus = origin

        for prop in proportions:
            line = Line(prop * usable_length, orientation, origin=terminus)
            lines.append(line)
            gap = Line(gap_length, orientation, origin=line.terminus)
            terminus = gap.terminus

        return lines


@dataclass(frozen=True)
class Path:
    """A closed polygonal path defined by a sequence of 2-D points.

    Parameters
    ----------
    points: list[tuple[float, float]], optional
        Ordered list of ``(x, y)`` vertices (default empty list).
    """

    points: list[tuple[float, float]] = field(default_factory=lambda: [])

    def to_svg(self) -> str:
        """Serialise the path as an SVG path ``d`` attribute string.

        Returns
        -------
        str
            An SVG path string starting with ``M``, joined by ``L`` segments,
            and closed with ``Z``.
        """
        path = "M "
        path += " L".join(f"{point[0]},{point[1]}" for point in self.points)
        path += " Z"

        return path

    @staticmethod
    def from_components(x: list[float], y: list[float]) -> 'Path':
        """Create a :class:`Path` from separate x and y coordinate lists.

        Parameters
        ----------
        x: list[float]
            x-coordinates of the vertices.
        y: list[float]
            y-coordinates of the vertices.

        Returns
        -------
        Path
            A :class:`Path` whose points are ``zip(x, y)``.
        """
        return Path(points=list(zip(x, y)))


@dataclass(frozen=True)
class Shape(ABC):
    """Abstract base class for geometric shapes that can be converted to a :class:`Path`."""

    @abstractmethod
    def to_path(self) -> Path:
        """Convert the shape to a :class:`Path` suitable for SVG rendering.

        Returns
        -------
        Path
            The closed polygonal approximation of the shape.
        """
        raise RuntimeError("Not meant to be instantiated")


@dataclass(frozen=True)
class ArcShape(Shape):
    """A filled arc shape with a given radius and thickness.

    Parameters
    ----------
    arc: Arc
        The angular extent of the shape.
    thickness: float
        Radial thickness of the filled band.
    radius: float
        Inner radius of the shape.
    """

    arc: Arc
    thickness: float
    radius: float

    @cached_property
    def interior_edge(self) -> np.ndarray:
        """Sampled points along the inner arc edge."""
        return self.arc.sample(self.radius)

    @cached_property
    def exterior_edge(self) -> np.ndarray:
        """Sampled points along the outer arc edge."""
        return self.arc.sample(self.radius + self.thickness)

    def to_path(self) -> Path:
        """Convert to a closed :class:`Path` by joining exterior and reversed interior edges."""
        x = self.exterior_edge.real.tolist() + self.interior_edge.real.tolist()[::-1]
        y = self.exterior_edge.imag.tolist() + self.interior_edge.imag.tolist()[::-1]
        return Path.from_components(x, y)


@dataclass(frozen=True)
class RibbonShape(Shape):
    """A filled ribbon bounded by two Bézier curves and two arc caps.

    Parameters
    ----------
    left_ribbon: Ribbon
        The left boundary ribbon (connects ``arc_source.begin`` to ``arc_target.begin``).
    right_ribbon: Ribbon
        The right boundary ribbon (connects ``arc_source.end`` to ``arc_target.end``).
    radius: float
        Circle radius for the ribbon endpoints.
    strength: float
        Bezier control-point strength for both ribbons.
    """

    left_ribbon: Ribbon
    right_ribbon: Ribbon
    radius: float
    strength: float

    def to_path(self) -> Path:
        """Convert to a closed :class:`Path` by joining both ribbon curves and arc caps.

        The path traces: left ribbon (Bezier) → arc cap at target end →
        right ribbon reversed (Bezier) → arc cap at source end (reversed).
        """
        left_samples = self.left_ribbon.sample(self.radius, strength=self.strength)
        right_samples = self.right_ribbon.sample(self.radius, strength=self.strength)

        cap_end = Arc(self.left_ribbon.end, self.right_ribbon.end).sample(self.radius)
        cap_begin = Arc(self.left_ribbon.begin, self.right_ribbon.begin).sample(self.radius)

        x = (left_samples[0].tolist() + cap_end.real.tolist()
             + right_samples[0].tolist()[::-1] + cap_begin.real.tolist()[::-1])
        y = (left_samples[1].tolist() + cap_end.imag.tolist()
             + right_samples[1].tolist()[::-1] + cap_begin.imag.tolist()[::-1])

        return Path.from_components(x, y)


@dataclass(frozen=True)
class LineShape(Shape):
    """A filled rectangular shape centred on a line segment.

    Parameters
    ----------
    line: Line
        The centreline of the shape.
    thickness: float
        Total perpendicular width of the shape.
    """

    line: Line
    thickness: float

    @cached_property
    def __half_thickness(self) -> float:
        """Half of the total thickness, used as the perpendicular offset for each edge."""
        return self.thickness / 2

    @cached_property
    def left_edge(self) -> np.ndarray:
        """Sampled start and end points of the left edge (negative offset side)."""
        return self.line.sample(-self.__half_thickness)

    @cached_property
    def right_edge(self) -> np.ndarray:
        """Sampled start and end points of the right edge (positive offset side)."""
        return self.line.sample(self.__half_thickness)

    def to_path(self) -> Path:
        """Convert to a closed :class:`Path` by joining left and reversed right edges."""
        x_left, y_left = self.left_edge.T
        x_right, y_right = self.right_edge.T
        x = x_left.tolist() + x_right.tolist()[::-1]
        y = y_left.tolist() + y_right.tolist()[::-1]
        return Path.from_components(x, y)
