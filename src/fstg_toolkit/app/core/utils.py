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
