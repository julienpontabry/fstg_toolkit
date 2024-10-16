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
    """
    Find the networks (connected components) per region in the input graph.

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
    """
    Compute the internal strength of a network in a graph.

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
    """
    Find the areas in the given networks that are adjacent in the connectivity graph.

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
    """
    Compute the correlation between networks.

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
    """
    Compute a graph of networks grouped by regions from an areas connectivity graph.

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
