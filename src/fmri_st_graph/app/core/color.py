from dataclasses import dataclass
from abc import ABC, abstractmethod
import colorsys


class ColorsInterpolator(ABC):
    @abstractmethod
    def sample(self, n: int) -> list[tuple[float, float, float]]:
        raise RuntimeError("Class not meant to be instantiated!")


@dataclass(frozen=True)
class HueInterpolator(ColorsInterpolator):
    saturation: float = 0.7
    value: float = 0.9

    def sample(self, n: int) -> list[tuple[float, float, float]]:
        colors = []

        for i in range(n):
            hue = i / n
            rgb = colorsys.hsv_to_rgb(hue, self.saturation, self.value)
            colors.append(rgb)

        return colors
