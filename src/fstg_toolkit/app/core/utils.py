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

from typing import Iterable
import re


def split_factors_from_name(names: Iterable[str],
                            pattern: re.Pattern = re.compile(r'([^_/]+)')) -> tuple[list[set], list[tuple[str, ...]]]:
    """Splits input names into factors and identifiers.

     The names are split into parts as specified by the provided regex pattern. By default,
     the parts are separated by underscores or slashes.

     The factors are parts changing across the names, while fixed parts are discarded.
     The identifiers defined by the factors with the most variability.

    Parameters
    ----------
    names : Iterable[str]
        An iterable of string names to be split.
    pattern : re.Pattern, optional
        A compiled regular expression pattern used to split the names (default is r'([^_/]+)').

    Returns
    -------
    factors : list of set
        List of sets, each containing unique factor values found at each position across all names.
    identifiers : list of tuple of str
        List of tuples, each tuple representing the identifiers for each name.

    Examples
    --------
    >>> factors, ids = split_factors_from_name(['a_b_c', 'a_b_d', 'a_x_c'])
    >>> factors == [{'c', 'd'}]
    True
    >>> ids
    [('c', 'b'), ('d', 'b'), ('c', 'x')]
    >>> factors, ids = split_factors_from_name(['foo_bar', 'foo_baz', 'foo_qux'])
    >>> factors == []
    True
    >>> ids
    [('bar',), ('baz',), ('qux',)]
    >>> factors, ids = split_factors_from_name(['2-3-1', '2-4-1', '5-7-1'], pattern=re.compile(r'([^-]+)'))
    >>> factors == [{'2', '5'}]
    True
    >>> ids
    [('2', '3'), ('2', '4'), ('5', '7')]
    """
    names_parts = [pattern.findall(name) for name in names]

    factors = []
    identifiers = []

    for parts in zip(*names_parts):
        s = set(parts)

        if len(s) > 1:
            factors.append(s)
            identifiers.append(parts)

    idx_ids = max(enumerate(factors), key=lambda s: len(s[1]))[0]
    del factors[idx_ids]

    ids = identifiers[idx_ids]
    del identifiers[idx_ids]
    identifiers.append(ids)

    return factors, list(zip(*identifiers))
