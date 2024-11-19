import re
from pathlib import Path
from typing import Optional

import click
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from fmri_st_graph import generate_pattern
from .factory import spatio_temporal_graph_from_corr_matrices
from .io import load_spatio_temporal_graph, save_spatio_temporal_graph
from .visualization import spatial_plot, temporal_plot, multipartite_plot, dynamic_plot


@click.group()
def cli():
    """Build, plot and simulate spatio-temporal graphs for fMRI data."""
    pass


## building ###################################################################

@cli.command()
@click.argument('correlation_matrices_path', type=click.Path(exists=True))
@click.argument('areas_description_path', type=click.Path(exists=True))
@click.option('-o', '--output_graph', type=click.Path(writable=True), default="st_graph.zip",
              help="Path where to write the built graph.")
def build(correlation_matrices_path: str, areas_description_path: str, output_graph: str):
    """Build a spatio-temporal graph.

    The spatio-temporal graph is built from correlation matrices and areas description.
    """
    matrices = np.load(correlation_matrices_path)

    if isinstance(matrices, np.lib.npyio.NpzFile):
        click.echo("The following sequences of matrices are available from the file:")
        click.echo(";\n".join(matrices.keys()) + ".")
        chosen = click.prompt("Which one to process ('all' to process all)?", default='all')

        if chosen == 'all':
            output_path = Path(output_graph)
            if not output_path.is_dir():
                click.echo("Output path must be a directory!", err=True)
                return
            matrices = [(matrices[k], output_path / f"{k}.zip") for k in matrices.keys()]
        else:
            matrices = [(matrices[chosen], output_graph)]
    else:
        matrices = [(matrices, output_graph)]

    areas = pd.read_csv(areas_description_path, index_col='Id_Area')

    with click.progressbar(matrices) as bar:
        for mat, output in bar:
            graph = spatio_temporal_graph_from_corr_matrices(mat, areas)
            save_spatio_temporal_graph(graph, output)


## plotting ###################################################################

@click.group()
@click.argument('graph_path', type=click.Path(exists=True))
@click.pass_context
def plot(ctx: click.core.Context, graph_path: str):
    """Plot a spatio-temporal graph."""
    ctx.obj = load_spatio_temporal_graph(graph_path)


@plot.command()
@click.pass_context
def multipartite(ctx: click.core.Context):
    """Plot as multipartite graph.

    The x-axis corresponds to the time evolution and nodes
    at a given time are lined up vertically.
    """
    go_on = True
    n = len(ctx.obj)
    if n >= 100:
        click.echo(f"There are {n} nodes in this graph; the multipartite plot has "
                   "not been optimized for big graphs!", err=True)
        answer = click.prompt("Do you really want to continue?",
                              type=click.Choice(["yes", "no"]), default="no")
        go_on = answer == 'yes'

    if go_on:
        fig, axe = plt.subplots(layout='constrained')
        multipartite_plot(ctx.obj, ax=axe)
        plt.show()


@plot.command()
@click.option('-t', '--time', type=click.IntRange(0), default=0,
              help="The time index of the spatial subgraph to show.")
@click.pass_context
def spatial(ctx: click.core.Context, time: int):
    """Plot as a spatial connectivity graph.

    The temporal edges are not displayed."""
    time = min(time, ctx.obj.graph['max_time'])
    fig, axe = plt.subplots(layout='constrained')
    spatial_plot(ctx.obj, time, ax=axe)
    plt.show()


@plot.command()
@click.pass_context
def temporal(ctx: click.core.Context):
    """Plot as a temporal connectivity graph.

    The spatial edges are not displayed."""
    fig, axe = plt.subplots(layout='constrained')
    temporal_plot(ctx.obj, ax=axe)
    plt.show()


@plot.command()
@click.option('-s', '--size', type=click.FloatRange(0), default=70,
              help="The size of the plotting window (in centimeter).")
@click.option('-w', '--window', type=click.IntRange(min=50), default=None,
              help="The size of the sliding temporal window to display (if not set, "
                   "show the full temporal plot by default).")
@click.pass_context
def dynamic(ctx: click.core.Context, size: float, window: int):
    """Plot a dynamic graph.

    Both the spatial and temporal graphs are drawn, with some interactivity."""
    dynamic_plot(ctx.obj, size, time_window=window)
    plt.show()


## simulating #################################################################

class GraphElementsDescription(click.ParamType):
    main_desc: str = r'[-,:.\d]+'
    elem_desc: str

    def __init__(self):
        self.main_regex = re.compile(self.main_desc)
        self.elem_regex = re.compile(self.elem_desc)

    def _convert_from_match(self, match: re.Match[str]) -> tuple[any,...]:
        pass

    def convert(self, value: any, param: Optional[click.Parameter], ctx: Optional[click.Context]) -> any:
        elements = []

        try:
            for main_match in self.main_regex.finditer(value):
                if not main_match:
                    self.fail(f"{value!r} is not a valid spatial edge description!")

                elem_match = self.elem_regex.match(main_match.group(0))

                if not elem_match:
                    self.fail(f"{value!r} is not a valid spatial edge description!")

                elements.append(self._convert_from_match(elem_match))
        except ValueError:
            self.fail(f"{value!r} is not a valid spatial edge description!")

        return elements


class NetworksDescription(click.ParamType):
    net_desc = re.compile(r'[-,:.\d]+')
    reg_desc = re.compile(r'\s*(?P<range>\d+:\d+),(?P<id>\d+),(?P<strength>-?\d*.\d+)')

    def __convert_network(self, value: str) -> list[tuple[tuple[int, int], int, float]]:
        network = []

        try:
            for net_match in self.net_desc.finditer(value):
                if not net_match:
                    self.fail(f"{value!r} is not a valid network description!")

                reg_match = self.reg_desc.match(net_match.group(0))

                if not reg_match:
                    self.fail(f"{value!r} is not a valid network description!")

                tmp = reg_match.group('range').split(':')
                areas = int(tmp[0]), int(tmp[1])
                region = int(reg_match.group('id'))
                internal_strength = float(reg_match.group('strength'))
                network.append((areas, region, internal_strength))

            return network
        except ValueError:
            self.fail(f"{value!r} is not a valid network description!")

    def convert(self, value: any, param: Optional[click.Parameter], ctx: Optional[click.Context]) -> any:
        if not isinstance(value, str):
            self.fail(f"{value!r} is not a valid networks description!")

        return [self.__convert_network(network_desc) for network_desc in value.split('/')]

NETWORKS_DESCRIPTION = NetworksDescription()


class SpatialEdgesDescription(GraphElementsDescription):#click.ParamType):
    elem_desc = r'(?P<n1>\d+),(?P<n2>\d+),(?P<corr>-?\d*.\d+)'

    def _convert_from_match(self, match: re.Match[str]) -> tuple[any,...]:
        node1 = int(match.group('n1'))
        node2 = int(match.group('n2'))
        correlation = float(match.group('corr'))
        return node1, node2, correlation

SPATIAL_EDGES_DESCRIPTION = SpatialEdgesDescription()


class TemporalEdgesDescription(GraphElementsDescription):
    main_desc = r'[-,.\w]+'
    elem_desc = r'(?P<n1>\d+(-\d+)?),(?P<n2>\d+),(?P<transition>\bsplit\b|\bmerge\b|\beq\b)'

    @staticmethod
    def __build_range(s: str) -> tuple[int, int] | int:
        if '-' in s:
            tmp = s.split('-')
            return int(tmp[0]), int(tmp[1])
        else:
            return int(s)

    def _convert_from_match(self, match: re.Match[str]) -> tuple[any,...]:
        transition = match.group('transition')

        if transition == 'eq':
            node1 = int(match.group('n1'))
            node2 = int(match.group('n2'))
        else:
            node1 = TemporalEdgesDescription.__build_range(match.group('n1'))
            node2 = TemporalEdgesDescription.__build_range(match.group('n2'))

        return node1, node2, transition

TEMPORAL_EDGES_DESCRIPTION = TemporalEdgesDescription()


@click.group()
def simulate():
    """Simulate a spatio-temporal graph."""
    pass

@simulate.command()
@click.argument('networks', type=NETWORKS_DESCRIPTION)
@click.argument('spatial_edges', type=SPATIAL_EDGES_DESCRIPTION, required=False)
@click.argument('temporal_edges', type=TEMPORAL_EDGES_DESCRIPTION, required=False)
@click.option('-o', '--output_path', type=click.Path(writable=True), default="pattern.zip",
              help="Path where to write the built graph.")
def pattern(networks: list[list[tuple[tuple[int, int], int, float]]],
            spatial_edges: list[tuple[int, int, float]] | None,
            temporal_edges: list[tuple[int, int, str]] | None,
            output_path: Path):
    pattern = generate_pattern(networks_list=networks,
                               spatial_edges=spatial_edges,
                               temporal_edges=temporal_edges)
    save_spatio_temporal_graph(pattern, output_path)


@simulate.command()
def graph():
    # TODO input patterns + sequence
    pass


@simulate.command()
def matrices():
    # TODO input graph + areas + threshold
    pass


if __name__ == '__main__':
    cli.add_command(plot)
    cli.add_command(simulate)
    cli()
