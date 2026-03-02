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

import colorsys
from abc import ABC, abstractmethod
from dataclasses import dataclass


class ColorsInterpolator(ABC):
    """Abstract base class for color interpolators that generate a palette of ``n`` colors."""

    @abstractmethod
    def sample(self, n: int) -> list[tuple[float, float, float]]:
        """Generate ``n`` evenly-distributed colors.

        Parameters
        ----------
        n: int
            Number of colors to generate.

        Returns
        -------
        list[tuple[float, float, float]]
            A list of ``n`` RGB tuples with components in [0, 1].
        """
        raise RuntimeError("Class not meant to be instantiated!")


@dataclass(frozen=True)
class HueInterpolator(ColorsInterpolator):
    """Color interpolator that steps evenly through the HSV hue wheel.

    Parameters
    ----------
    saturation: float, optional
        HSV saturation for all generated colors (default 0.7).
    value: float, optional
        HSV value (brightness) for all generated colors (default 0.9).
    """

    saturation: float = 0.7
    value: float = 0.9

    def sample(self, n: int) -> list[tuple[float, float, float]]:
        """Generate ``n`` colors by stepping uniformly through the hue wheel.

        Parameters
        ----------
        n: int
            Number of colors to generate.

        Returns
        -------
        list[tuple[float, float, float]]
            A list of ``n`` RGB tuples with components in [0, 1].
        """
        colors = []

        for i in range(n):
            hue = i / n
            rgb = colorsys.hsv_to_rgb(hue, self.saturation, self.value)
            colors.append(rgb)

        return colors
