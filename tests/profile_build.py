# Copyright 2025 ICube (University of Strasbourg - CNRS)
# author: Julien PONTABRY (ICube)
#
# This software is a computer program whose purpose is to provide a toolkit
# to model, process and analyze the longitudinal reorganization of brain
# connectivity data, as functional MRI for instance.
#
# This software is governed by the CeCILL-B license under French law and
# abiding by the rules of distribution of free software. You can use,
# modify and/or redistribute the software under the terms of the CeCILL-B
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info".
#
# As a counterpart to the access to the source code and rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty and the software's author, the holder of the
# economic rights, and the successive licensors have only limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading, using, modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean that it is complicated to manipulate, and that also
# therefore means that it is reserved for developers and experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and, more generally, to use and operate it in the
# same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL-B license and that you accept its terms.

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
