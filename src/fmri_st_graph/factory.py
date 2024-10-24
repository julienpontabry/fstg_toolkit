"""Defines helpers to create spatio-temporal graphs."""

from typing import Iterable

import networkx as nx
import numpy as np
import pandas as pd

from .graph import RC5, SpatioTemporalGraph


def graph_from_corr_matrix(matrix: np.array, areas_desc: pd.DataFrame, corr_thr: float = 0.4,
                           abs_thr: bool = True, area_col_name: str = 'Name_Area',
                           region_col_name: str = 'Name_Region') -> nx.Graph:
    """Compute a symmetric graph from a symmetric correlation matrix.

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

    Example
    -------
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


def __find_networks_per_region(graph: nx.Graph, regions: list[str]) -> list[tuple[str, set[int]]]:
    """Find the networks (connected components) per region in the input graph.

    Parameters
    ----------
    graph: networkx.Graph
        A symmetric graph.
    regions: list[str]
        A list of regions.

    Returns
    -------
    list[tuple[str, set[int]]]
        A list of tuples, each containing a region name and a set of connected nodes.
    """
    networks_nodes = []

    for region in regions:
        nodes_in_region = [node
                           for node, data in graph.nodes.items()
                           if data['region'] == region]
        region_subgraph = graph.subgraph(nodes_in_region)
        networks_nodes += list(map(lambda nodes: (region, nodes),
                                   nx.connected_components(region_subgraph)))

    return networks_nodes


def __compute_network_internal_strength(graph: nx.Graph, network: set[int]) -> float:
    """Compute the internal strength of a network in a graph.

    Parameters
    ----------
    graph: networkx.Graph
        The non-directed graph that contains the network.
    network: set[int]
        The network as a set of nodes.

    Returns
    -------
    float
        The internal strength, calculated as the average correlation in the network.
    """
    internal_correlations = [data['correlation']
                             for _, data in graph.subgraph(network).edges.items()]
    return 1 if len(internal_correlations) == 0 else float(np.mean(internal_correlations))


def __find_adjacent_areas(graph: nx.Graph, network1: set[int], network2: set[int]) -> list[tuple[int, int]]:
    """Find the areas in the given networks that are adjacent in the connectivity graph.

    Parameters
    ----------
    graph: networkx.Graph
        The non-directed connectivity graph.
    network1: set[int]
        A network from which one area is selected to test for connectivity.
    network2: set[int]
        Another network from which another area is selected to test for connectivity.

    Returns
    -------
    list[tuple[int, int]]
        A list of couples of area, one from each of the input network, that are connected in the graph.
    """
    return [(node1, node2)
            for node1 in network1
            for node2 in network2
            if graph.has_edge(node1, node2)]


def __compute_correlation_between_networks(graph: nx.Graph, adjacent_areas: list[tuple[int, int]]) -> float:
    """Compute the correlation between networks.

    Parameters
    ----------
    graph: networkx.Graph
        The non-directed graph that contains the networks.
    adjacent_areas: list[tuple[int, int]]
        A list of adjacent areas from two networks.

    Returns
    -------
    float
        The maximal correlation of all edges that connect two areas from two networks.
    """
    correlations: list[float] = [graph[area1][area2]['correlation'] for area1, area2 in adjacent_areas]
    return correlations[np.argmax(np.abs(correlations))]


def networks_from_connect_graph(graph: nx.Graph, regions: list[str]) -> nx.Graph:
    """Compute a graph of networks grouped by regions from an areas connectivity graph.

    The networks are groups of areas that forms connected components within a region.

    Parameters
    ----------
    graph: networkx.Graph
        The areas connectivity graph.
    regions: list[str]
        A list of regions' names.

    Returns
    -------
    networkx.Graph
        A graph of areas in networks.

    Example
    -------
    Let be a connectivity graph between areas, like described below.
    >>> G = nx.Graph()
    >>> G.add_nodes_from([
    ...    (1, {'area': 'A1', 'region': 'R1'}),
    ...    (2, {'area': 'A2', 'region': 'R1'}),
    ...    (3, {'area': 'A3', 'region': 'R2'})])
    >>> G.add_edges_from([
    ...     (1, 3, {'correlation': -0.41853318}),
    ...     (2, 3, {'correlation': 0.75087697})])

    The graph of networks of connected areas for both regions is built using the following.
    >>> netG = networks_from_connect_graph(G, ['R1', 'R2'])
    >>> netG.nodes(data=True)
    NodeDataView({1: {'areas': {1}, 'region': 'R1', 'internal_strength': 1}, 2: {'areas': {2}, 'region': 'R1', 'internal_strength': 1}, 3: {'areas': {3}, 'region': 'R2', 'internal_strength': 1}})
    >>> netG.edges(data=True)
    EdgeDataView([(1, 3, {'correlation': -0.41853318}), (2, 3, {'correlation': 0.75087697})])
    """
    networks = __find_networks_per_region(graph, regions)
    networks_graph = nx.Graph()

    for i, (region, network) in enumerate(networks):
        internal_strength = __compute_network_internal_strength(graph, network)
        networks_graph.add_node(i+1, areas=network, region=region,
                                internal_strength=internal_strength)

        for j, (_, other_network) in enumerate(networks[:i]):
            adjacent_areas = __find_adjacent_areas(graph, network, other_network)

            if len(adjacent_areas) > 0:
                correlation = __compute_correlation_between_networks(graph, adjacent_areas)
                networks_graph.add_edge(i + 1, j + 1, correlation=correlation)

    return networks_graph


def __find_temporal_transition(network1: set[int], network2: set[int]) -> RC5:
    """Find an RC5 temporal transition between two successive networks in time.

    Parameters
    ----------
    network1: set[int]
        A network at time t.
    network2: set[int]
        Another network at time t+1.

    Returns
    -------
    RC5
        The RC5 transition between the networks.
    """
    if len(network1 & network2) == 0:
        return RC5.DC
    else:
        if network1 <= network2:
            return RC5.EQ if network1 >= network2 else RC5.PP
        else:
            return RC5.PPi if network1 >= network2 else RC5.PO


def __add_networks_graph(st_graph: nx.DiGraph, networks_graph: nx.Graph, time: int) -> None:
    """Add the networks graph at time point to the spatio-temporal graph.

    Parameters
    ----------
    st_graph: networkx.DiGraph
        The spatio-temporal graph.
    networks_graph: networkx.Graph
        The networks graph.
    time: int
        The time point index of the networks graph.
    """
    nb_nodes = len(st_graph)

    for node, data in networks_graph.nodes.items():
        st_graph.add_node(nb_nodes + node, t=time, **data)

    # add twice the spatial edges because the ST-graph is directed
    for (node1, node2), data in networks_graph.edges.items():
        new_node1 = nb_nodes + node1
        new_node2 = nb_nodes + node2
        st_graph.add_edge(new_node1, new_node2, t=time, type='spatial', **data)
        st_graph.add_edge(new_node2, new_node1, t=time, type='spatial', **data)


def spatio_temporal_graph_from_networks_graphs(networks_graphs: tuple[nx.Graph, ...]) -> nx.DiGraph:
    """Compute a spatio-temporal graph that encompasses networks graphs at successive time points.

    Parameters
    ----------
    networks_graphs: tuple[networkx.Graph]
        The networks graphs at successive time points.

    Returns
    -------
    networkx.DiGraph
        The spatio-temporal graph.

    Example
    -------
    >>> netG1 = nx.Graph()
    >>> netG1.add_nodes_from([
    ...     (1, {'areas': {1}, 'region': 'R1', 'internal_strength': 1}),
    ...     (2, {'areas': {2, 3}, 'region': 'R2', 'internal_strength': -0.5})])
    >>> netG1.add_edges_from([(1, 2, {'correlation': -0.42})])
    >>> netG2 = nx.Graph()
    >>> netG2.add_nodes_from([
    ...     (1, {'areas': {1, 2}, 'region': 'R1', 'internal_strength': 0.75}),
    ...     (2, {'areas': {3}, 'region': 'R2', 'internal_strength': 1})])
    >>> netG2.add_edges_from([(1, 2, {'correlation': 0.41})])
    >>> st_G = spatio_temporal_graph_from_networks_graphs((netG1, netG2))
    >>> st_G.nodes(data=True)
    NodeDataView({1: {'t': 0, 'areas': {1}, 'region': 'R1', 'internal_strength': 1},
                  2: {'t': 0, 'areas': {2, 3}, 'region': 'R2', 'internal_strength': -0.5},
                  3: {'t': 1, 'areas': {1, 2}, 'region': 'R1', 'internal_strength': 0.75},
                  4: {'t': 1, 'areas': {3}, 'region': 'R2', 'internal_strength': 1}})
    >>> st_G.edges(data=True)
    OutEdgeDataView([(1, 2, {'t': 0, 'type': 'spatial', 'correlation': -0.42}),
                     (1, 3, {'type': 'temporal', 'transition': <RC5.PP: 2>}),
                     (2, 1, {'t': 0, 'type': 'spatial', 'correlation': -0.42}),
                     (2, 3, {'type': 'temporal', 'transition': <RC5.PO: 4>}),
                     (2, 4, {'type': 'temporal', 'transition': <RC5.PPi: 3>}),
                     (3, 4, {'t': 1, 'type': 'spatial', 'correlation': 0.41}),
                     (4, 3, {'t': 1, 'type': 'spatial', 'correlation': 0.41})])
    """
    nb_networks_graphs = len(networks_graphs)
    st_graph = nx.DiGraph(min_time=0, max_time=nb_networks_graphs)
    __add_networks_graph(st_graph, networks_graphs[0], 0)
    prev_node = 0

    for t in range(nb_networks_graphs - 1):
        networks_graph_t0 = networks_graphs[t]
        networks_graph_t1 = networks_graphs[t + 1]

        __add_networks_graph(st_graph, networks_graph_t1, t + 1)

        cur_node = prev_node  # initialize variable in case there is no nodes in t0 graph
        for node_t0, data_t0 in networks_graph_t0.nodes.items():
            cur_node = prev_node + len(networks_graph_t0)

            # filter by regions?
            for n_t1, d_t1 in networks_graph_t1.nodes.items():
                transition = __find_temporal_transition(data_t0['areas'], d_t1['areas'])

                if transition != RC5.DC:
                    st_graph.add_edge(prev_node + node_t0, cur_node + n_t1,
                                      type='temporal', transition=transition)

        prev_node = cur_node

    return st_graph


def spatio_temporal_graph_from_corr_matrices(corr_matrices: Iterable[np.array], areas_desc: pd.DataFrame,
                                             region_col_name: str = 'Name_Region',
                                             **corr_mat_kwd) -> SpatioTemporalGraph:
    """Build a spatio-temporal graph a set of temporal correlation matrices and areas description.

    Parameters
    ----------
    corr_matrices: numpy.array
        The successive correlation matrices (one per time point).
    areas_desc: pandas.DataFrame
        The description of areas to consider.
    region_col_name: str, optional
        The name of the column in areas_desc that gives the associated region name (default
        is 'Name_Region').
    corr_mat_kwd: dict, optional
        The options for the construction of a connectivity graph from a correlation matrix.
        For more information, see :func:`~factory.graph_from_corr_matrix`.

    Returns
    -------
    SpatioTemporalGraph
        A structure that hold a spatio-temporal graph of functional connectivity data.

    Example
    -------
    >>> M1 = np.array([
    ...     [          1,  0.12873788, -0.41853318],
    ...     [ 0.12873788,           1,  0.75087697],
    ...     [-0.41853318,  0.75087697,           1]])
    >>> M2 = np.array([
    ...     [          1,  0.52873788, -0.41853318],
    ...     [ 0.52873788,           1,  0.75087697],
    ...     [-0.41853318,  0.75087697,           1]])
    >>> areas = pd.DataFrame({'Id': [1, 2, 3], 'Name_Area': ['A1', 'A2', 'A3'], 'Name_Region': ['R1', 'R1', 'R2']})
    >>> areas.set_index('Id', inplace=True)
    >>> result = spatio_temporal_graph_from_corr_matrices((M1, M2), areas)
    >>> result.nodes(data=True)
    NodeDataView({1: {'t': 0, 'areas': {1}, 'region': 'R1', 'internal_strength': 1},
                  2: {'t': 0, 'areas': {2}, 'region': 'R1', 'internal_strength': 1},
                  3: {'t': 0, 'areas': {3}, 'region': 'R2', 'internal_strength': 1},
                  4: {'t': 1, 'areas': {1, 2}, 'region': 'R1', 'internal_strength': 0.52873788},
                  5: {'t': 1, 'areas': {3}, 'region': 'R2', 'internal_strength': 1}})
    >>> result.edges(data=True)
    OutEdgeDataView([(1, 3, {'t': 0, 'type': 'spatial', 'correlation': -0.41853318}),
                     (1, 4, {'type': 'temporal', 'transition': <RC5.PP: 2>}),
                     (2, 3, {'t': 0, 'type': 'spatial', 'correlation': 0.75087697}),
                     (2, 4, {'type': 'temporal', 'transition': <RC5.PP: 2>}),
                     (3, 1, {'t': 0, 'type': 'spatial', 'correlation': -0.41853318}),
                     (3, 2, {'t': 0, 'type': 'spatial', 'correlation': 0.75087697}),
                     (3, 5, {'type': 'temporal', 'transition': <RC5.EQ: 1>}),
                     (4, 5, {'t': 1, 'type': 'spatial', 'correlation': 0.75087697}),
                     (5, 4, {'t': 1, 'type': 'spatial', 'correlation': 0.75087697})])
    """
    regions = areas_desc[region_col_name].unique()
    graph =  spatio_temporal_graph_from_networks_graphs(tuple(
        networks_from_connect_graph(
            graph_from_corr_matrix(corr_matrix, areas_desc, region_col_name=region_col_name, **corr_mat_kwd),
            regions)
        for corr_matrix in corr_matrices))
    return SpatioTemporalGraph(graph, areas_desc)
