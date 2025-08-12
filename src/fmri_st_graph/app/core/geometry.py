from dataclasses import dataclass
from functools import cached_property

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
