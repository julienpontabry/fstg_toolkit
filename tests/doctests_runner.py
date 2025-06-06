if __name__ == '__main__':
    import doctest

    from fmri_st_graph import graph, factory, io, simulation
    doctest.testmod(graph, optionflags=doctest.NORMALIZE_WHITESPACE)
    doctest.testmod(factory, optionflags=doctest.NORMALIZE_WHITESPACE)
    doctest.testmod(io, optionflags=doctest.NORMALIZE_WHITESPACE)
    doctest.testmod(simulation, optionflags=doctest.NORMALIZE_WHITESPACE)

    from app.core import utils
    doctest.testmod(utils, optionflags=doctest.NORMALIZE_WHITESPACE)
