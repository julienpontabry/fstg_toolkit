"""Defines helpers to create spatio-temporal graphs."""

import numpy as np
import networkx as nx
import pandas as pd


def graph_from_corr_matrix(matrix: np.array, areas_desc: pd.DataFrame, corr_thr: float = 0.4,
                           abs_thr: bool = True, area_col_name: str = 'Name_Area',
                           region_col_name: str = 'Name_Region') -> nx.Graph:
    """
    Compute a symmetric graph from a symmetric correlation matrix.

    Parameters
    ----------
    matrix: numpy.array
        The symmetric correlation matrix.
    areas_desc: pandas.DataFrame
        The dataframe listing the areas.
    corr_thr: float, optional
        The threshold above which correlations are taken into account (default is 0.4).
    abs_thr: bool, optional
        flag to use the threshold on absolute correlations (default is true).
    area_col_name: str, optional
        The name of the column of areas' names (default is 'Name_Area').
    region_col_name: str, optional
        The name of the column of regions' names (default is 'Name_Region').

    Returns
    -------
    networkx.Graph
        A non-directed graph with related area and region carried in node's data and
        correlation carried in edge's data.

    Examples
    --------
    One needs a symmetric matrix of correlation and a dataframe of areas with indices starting
    at 1 and names of areas and regions, as below.
    >>> M = np.array([
    ...     [          1,  0.12873788, -0.41853318],
    ...     [ 0.12873788,           1,  0.75087697],
    ...     [-0.41853318,  0.75087697,           1]])
    >>> areas = pd.DataFrame({'Id': [1, 2, 3], 'Name_Area': ['A1', 'A2', 'A3'], 'Name_Region': ['R1', 'R1', 'R2']})
    >>> areas.set_index('Id', inplace=True)

    The matrix needs to be symmetric because the function outputs a non-directed graph, so it
    cannot handle properly asymmetric matrices. Also, its diagonal should be filled by ones,
    but it is not mandatory, as the function does not read it.

    Finally, with the previous input (and defaults optional parameters), the function outputs
    the following non-directed graph.
    >>> G = graph_from_corr_matrix(M, areas)
    >>> G.nodes(data=True)
    NodeDataView({1: {'area': 'A1', 'region': 'R1'}, 2: {'area': 'A2', 'region': 'R1'}, 3: {'area': 'A3', 'region': 'R2'}})
    >>> G.edges(data=True)
    EdgeDataView([(1, 3, {'correlation': -0.41853318}), (2, 3, {'correlation': 0.75087697})])
    """
    corr_trans = abs if abs_thr else lambda x: x
    graph = nx.Graph()

    for i in range(len(matrix)):
        area_desc = areas_desc.loc[i+1]
        graph.add_node(i+1, area=area_desc[area_col_name], region=area_desc[region_col_name])

        for j in range(i):
            corr = float(matrix[i, j])

            if corr_trans(corr) > corr_thr:
                graph.add_edge(i+1, j+1, correlation=corr)

    return graph
