if __name__ == '__main__':
    import fmri_st_graph as fg

    graph = fg.load_spatio_temporal_graph('/tmp/st_graph.zip')
    fg.spatial_plot(graph, 0)
    fg.temporal_plot(graph)
