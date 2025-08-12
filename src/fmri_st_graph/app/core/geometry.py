from dataclasses import dataclass, field
from functools import cached_property, cache

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
class ArcShape:
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
