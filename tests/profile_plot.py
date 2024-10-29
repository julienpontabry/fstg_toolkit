if __name__ == '__main__':
    import timeit
    import fmri_st_graph as fg

    graph = fg.load_spatio_temporal_graph('/tmp/st_graph.zip')
    fg.spatial_plot(graph, 0)

    timer = timeit.Timer(lambda: fg.temporal_plot(graph))
    n, _ = timer.autorange()
    r = 5
    print(f"Min. time is {round(min(timer.repeat(r, n)) * 1_000)}ms (over {r} repeats).")
