from pathlib import Path
import timeit
import fstg_toolkit as fg
from fstg_toolkit.io import DataLoader


if __name__ == '__main__':
    loader = DataLoader(Path('/tmp/st_graph.zip'))
    areas = loader.load_areas()
    filenames = loader.lazy_load_graphs()
    graph = loader.load_graph(areas, filenames[0])

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
