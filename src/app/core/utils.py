from typing import Iterable
import re


def split_factors_from_name(names: Iterable[str],
                            pattern: re.Pattern = re.compile(r'([^_/]+)')) -> tuple[list[set], list[tuple[str, ...]]]:
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
