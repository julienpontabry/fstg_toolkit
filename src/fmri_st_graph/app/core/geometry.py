from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from functools import cached_property
from math import cos, sin

import numpy as np
from numpy import pi


@dataclass(frozen=True)
class Arc:
    begin: float
    end: float

    @cached_property
    def angle(self) -> float:
        return self.end - self.begin

    def sample(self, radius: float, angular_res: float = pi/64) -> np.ndarray:
        n_samples = max(int(round(self.angle / angular_res)), 2)
        theta = np.linspace(self.begin, self.end, n_samples)
        return radius * np.exp(1j * theta)

    @staticmethod
    def from_proportions(proportions: list[float], gap_size: float = 0.005) -> list['Arc']:
        gap = 2 * pi * gap_size
        arcs = []
        begin = 0

        for prop in proportions:
            end = begin + prop * 2*pi - gap
            arcs.append(Arc(begin, end))
            begin = end + gap

        return arcs


@dataclass(frozen=True)
class Line:
    length: float
    orientation: float
    origin: tuple[float, float] = (0, 0)

    @cached_property
    def terminus(self) -> tuple[float, float]:
        x = self.length * cos(self.orientation) + self.origin[0]
        y = self.length * sin(self.orientation) + self.origin[1]
        return x, y

    def sample(self, offset: float = 0) -> np.ndarray:
        # FIXME precalculate cos and sin
        ox = -offset * sin(self.orientation) + self.origin[0]
        oy =  offset * cos(self.orientation) + self.origin[1]
        x = self.length * cos(self.orientation) - offset * sin(self.orientation) + self.origin[0]
        y = self.length * sin(self.orientation) + offset * cos(self.orientation) + self.origin[1]
        return np.array([(ox, oy), (x, y)])

    @staticmethod
    def from_proportions(proportions: list[float], total_length: float, orientation: float,
                         origin: tuple[float, float] = (0, 0), gap_size: float = 0.005) -> list['Line']:
        lines = []
        gap_length = total_length * gap_size
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
    points: list[tuple[float, float]] = field(default_factory=lambda: [])

    def to_svg(self) -> str:
        path = "M "
        path += " L".join(f"{point[0]},{point[1]}" for point in self.points)
        path += " Z"

        return path

    @staticmethod
    def from_components(x: list[float], y: list[float]) -> 'Path':
        return Path(points=list(zip(x, y)))


@dataclass(frozen=True)
class Shape(ABC):
    @abstractmethod
    def to_path(self) -> Path:
        raise RuntimeError("Not meant to be instantiated")


@dataclass(frozen=True)
class ArcShape(Shape):
    arc: Arc
    thickness: float
    radius: float

    @cached_property
    def interior_edge(self) -> np.ndarray:
        return self.arc.sample(self.radius)

    @cached_property
    def exterior_edge(self) -> np.ndarray:
        return self.arc.sample(self.radius + self.thickness)

    def to_path(self) -> Path:
        x = self.exterior_edge.real.tolist() + self.interior_edge.real.tolist()[::-1]
        y = self.exterior_edge.imag.tolist() + self.interior_edge.imag.tolist()[::-1]
        return Path.from_components(x, y)

@dataclass(frozen=True)
class LineShape(Shape):
    line: Line
    thickness: float

    @cached_property
    def __half_thickness(self) -> float:
        return self.thickness / 2

    @cached_property
    def left_edge(self) -> np.ndarray:
        return self.line.sample(-self.__half_thickness)

    @cached_property
    def right_edge(self) -> np.ndarray:
        return self.line.sample(self.__half_thickness)

    def to_path(self) -> Path:
        x_left, y_left = self.left_edge.T
        x_right, y_right = self.right_edge.T
        x = x_left.tolist() + x_right.tolist()[::-1]
        y = y_left.tolist() + y_right.tolist()[::-1]
        return Path.from_components(x, y)
