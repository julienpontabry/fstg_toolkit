def build_st_graph(matrices, areas):
    return fg.spatio_temporal_graph_from_corr_matrices(matrices, areas)


if __name__ == '__main__':
    import timeit
    import numpy as np
    import pandas as pd
    import fstg_toolkit as fg

    correlation_matrices_path = "/home/jpontabry/Documents/projets/visualisation graphes spatio-temporels/data/list_of_corr_matrices_13months.zip"
    matrices = np.load(correlation_matrices_path)

    if isinstance(matrices, np.lib.npyio.NpzFile):
        matrices = matrices[list(matrices.keys())[0]]

    areas_description_path = "/home/jpontabry/Documents/projets/visualisation graphes spatio-temporels/data/brain_areas_regions_rel_full.csv"
    areas = pd.read_csv(areas_description_path, index_col='Id_Area')

    timer = timeit.Timer(lambda: build_st_graph(matrices, areas))
    n, _ = timer.autorange()
    r = 10
    print(f"Min. time is {round(min(timer.repeat(r, n)) * 1_000)}ms (over {r} repeats).")
