if __name__ == '__main__':
    import doctest

    from fstg_toolkit import graph, factory, io, simulation
    from fstg_toolkit.app.core import utils

    total_failed = 0

    for module in (graph, factory, io, simulation, utils):
        result = doctest.testmod(module, optionflags=doctest.NORMALIZE_WHITESPACE)
        total_failed += result.failed

    if total_failed > 0:
        exit(1)
