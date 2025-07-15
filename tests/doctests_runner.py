if __name__ == '__main__':
    import doctest

    from fmri_st_graph import graph, factory, io, simulation
    from fmri_st_graph.app.core import utils

    total_failed = 0

    for module in (graph, factory, io, simulation, utils):
        result = doctest.testmod(module, optionflags=doctest.NORMALIZE_WHITESPACE)
        total_failed += result.failed

    if total_failed > 0:
        exit(1)
