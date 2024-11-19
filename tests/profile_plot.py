if __name__ == '__main__':
    import timeit
    import fmri_st_graph as fg

    graph = fg.load_spatio_temporal_graph('/tmp/st_graph.zip')

    timer = timeit.Timer(lambda: fg.spatial_plot(graph, 0))
    n, _ = timer.autorange()
    print(n)
    r = 5
    print(f"Min. time for spatial plot is {round(min(timer.repeat(r, n)) * 1_000)}ms (over {r} repeats).")

    timer = timeit.Timer(lambda: fg.temporal_plot(graph))
    n, _ = timer.autorange()
    print(n)
    print(f"Min. time for temporal plot is {round(min(timer.repeat(r, n)) * 1_000)}ms (over {r} repeats).")

    timer = timeit.Timer(lambda: fg.multipartite_plot(graph))
    n, _ = timer.autorange()
    print(n)
    print(f"Min. time for multipartite plot is {round(min(timer.repeat(r, n)) * 1_000)}ms (over {r} repeats).")
