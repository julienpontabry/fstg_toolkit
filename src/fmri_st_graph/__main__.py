import re
from pathlib import Path
from typing import Optional

import click
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from fmri_st_graph import generate_pattern, SpatioTemporalGraphSimulator, CorrelationMatrixSequenceSimulator
from .factory import spatio_temporal_graph_from_corr_matrices
from .io import load_spatio_temporal_graph, save_spatio_temporal_graph
from .visualization import spatial_plot, temporal_plot, multipartite_plot, DynamicPlot


@click.group()
def cli():
    """Build, plot and simulate spatio-temporal graphs for fMRI data."""
    pass


## building ###################################################################

@cli.command()
@click.argument('correlation_matrices_path', type=click.Path(exists=True))
@click.argument('areas_description_path', type=click.Path(exists=True))
@click.option('-o', '--output_graph', type=click.Path(writable=True),
              default="output", show_default="a graph archive named 'output.zip' in the current directory",
              help="Path where to write the built graph. If there is no '.zip' extension, it will be added.")
def build(correlation_matrices_path: str, areas_description_path: str, output_graph: str):
    """Build a spatio-temporal graph from correlations matrices.

    The spatio-temporal graph written at OUTPUT will be built from correlations matrices in
    CORRELATION_MATRICES_PATH and areas description in AREAS_DESCRIPTION_PATH.

    If the file CORRELATION_MATRICES_PATH contains more than one set of matrices, the list of available
    names will be displayed and which one to use will be prompted.
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
    """Plot a spatio-temporal graph.

    The file GRAPH_PATH must be an archive containing a spatio-temporal graph.
    """
    ctx.obj = load_spatio_temporal_graph(graph_path)


@plot.command()
@click.pass_context
def multipartite(ctx: click.core.Context):
    """Plot as multipartite graph.

    The x-axis corresponds to the time evolution, and nodes
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
@click.option('-t', '--time', type=click.IntRange(0),
              default=0, show_default=True,
              help="The time index of the spatial subgraph to show.")
@click.pass_context
def spatial(ctx: click.core.Context, time: int):
    """Plot as a spatial connectivity graph.

    Only the nodes and spatial edges at a given time will be displayed.
    """
    max_time = ctx.obj.graph['max_time']
    if time > max_time:
        click.echo(f"The graph as a maximum time of {max_time} and "
                   f"requested time is greater ({time}>{max_time})!")
    else:
        time = min(time, max_time)
        fig, axe = plt.subplots(layout='constrained')
        spatial_plot(ctx.obj, time, ax=axe)
        plt.show()


@plot.command()
@click.pass_context
def temporal(ctx: click.core.Context):
    """Plot as a temporal connectivity graph.

    All the nodes and only the temporal edges will be displayed.
    """
    fig, axe = plt.subplots(layout='constrained')
    temporal_plot(ctx.obj, ax=axe)
    plt.show()


@plot.command()
@click.option('-s', '--size', type=click.FloatRange(0),
              default=70, show_default=True,
              help="The size of the plotting window (in centimeter).")
@click.pass_context
def dynamic(ctx: click.core.Context, size: float):
    """Plot in a dynamic graph.

    Both the spatial and temporal graphs will be displayed,
    with some interactivity.
    """
    DynamicPlot(ctx.obj, size).plot()
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
                    self.fail(f"{value!r} is not a valid description!")

                elem_match = self.elem_regex.match(main_match.group(0))

                if not elem_match:
                    self.fail(f"{value!r} is not a valid description!")

                elements.append(self._convert_from_match(elem_match))
        except ValueError:
            self.fail(f"{value!r} is not a valid description!")

        return elements


class _NetworkDescription(GraphElementsDescription):
    elem_desc = r'\s*(?P<range>\d+:\d+),(?P<id>\d+),(?P<strength>-?\d*.\d+)'

    def _convert_from_match(self, match: re.Match[str]) -> tuple[any, ...]:
        tmp = match.group('range').split(':')
        areas = int(tmp[0]), int(tmp[1])
        region = int(match.group('id'))
        internal_strength = float(match.group('strength'))
        return areas, region, internal_strength


class NetworksDescription(click.ParamType):
    _delegate = _NetworkDescription()

    def convert(self, value: any, param: Optional[click.Parameter], ctx: Optional[click.Context]) -> any:
        if not isinstance(value, str):
            self.fail(f"{value!r} is not a valid networks description!")
        return [self._delegate.convert(network_desc, param, ctx)
                for network_desc in value.split('/')]

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
    elem_desc = r'(?P<n1>\d+(-\d+)?),(?P<n2>\d+(-\d+)?)'

    @staticmethod
    def __build_range(s: str) -> tuple[int, int] | int:
        if '-' in s:
            tmp = s.split('-')
            return int(tmp[0]), int(tmp[1])
        else:
            return int(s)

    def _convert_from_match(self, match: re.Match[str]) -> tuple[any,...]:
        node1 = TemporalEdgesDescription.__build_range(match.group('n1'))
        node2 = TemporalEdgesDescription.__build_range(match.group('n2'))

        node1_is_int = isinstance(node1, int)
        node2_is_int = isinstance(node2, int)

        if node1_is_int and node2_is_int:
            transition = 'eq'
        elif node1_is_int:
            transition = 'split'
        elif node2_is_int:
            transition = 'merge'
        else:
            self.fail(f"Unsupported transition between nodes {match.group('n1')} and {match.group('n2')}!")

        return node1, node2, transition

TEMPORAL_EDGES_DESCRIPTION = TemporalEdgesDescription()


class GraphSequenceDescription(click.ParamType):
    def convert(self, value: any, param: Optional[click.Parameter], ctx: Optional[click.Context]) -> any:
        if not isinstance(value, str):
            self.fail(f"{value!r} is not a valid description!")

        elements = value.split(' ')
        sequence = []
        for element in elements:
            if element.isnumeric():
                sequence.append(int(element))
            elif element.lower().startswith('p') and element[1:].isnumeric():
                sequence.append(element.lower())
            else:
                self.fail(f"{element!r} is not a valid description for a pattern or a spacing!")

        return sequence


GRAPH_SEQUENCE_DESCRIPTION = GraphSequenceDescription()


@click.group()
@click.option('-o', '--output_path', type=click.Path(writable=True),
              default="output", show_default="a file named 'output' in the current directory",
              help="Path where to write the simulated output.")
@click.pass_context
def simulate(ctx: click.core.Context, output_path: Path):
    """Simulate a spatio-temporal graph."""
    ctx.obj = output_path

@simulate.command()
@click.argument('networks', type=NETWORKS_DESCRIPTION)
@click.argument('spatial_edges', type=SPATIAL_EDGES_DESCRIPTION, required=False)
@click.argument('temporal_edges', type=TEMPORAL_EDGES_DESCRIPTION, required=False)
@click.pass_context
def pattern(ctx: click.core.Context, networks: list[list[tuple[tuple[int, int], int, float]]],
            spatial_edges: list[tuple[int, int, float]] | None,
            temporal_edges: list[tuple[int, int, str]] | None):
    """Generate a spatio-temporal graph pattern from description."""
    pat = generate_pattern(networks_list=networks,
                           spatial_edges=spatial_edges,
                           temporal_edges=temporal_edges)
    save_spatio_temporal_graph(pat, ctx.obj)


@simulate.command()
@click.argument('patterns', nargs=-1, type=click.Path(exists=True))
@click.argument('sequence_description', type=GRAPH_SEQUENCE_DESCRIPTION)
@click.pass_context
def sequence(ctx: click.core.Context, patterns: tuple[Path], sequence_description: list[str | int]):
    """Generate a spatio-temporal graph from a patterns sequence."""
    pattern_graphs = {f'p{i+1}': load_spatio_temporal_graph(filepath)
                      for i, filepath in enumerate(patterns)}
    simulator = SpatioTemporalGraphSimulator(**pattern_graphs)
    graph = simulator.simulate(*sequence_description)
    save_spatio_temporal_graph(graph, ctx.obj)


@simulate.command()
@click.argument('graph_path', type=click.Path(exists=True))
@click.option('-t', '--threshold', type=click.FloatRange(0, 1, min_open=True, max_open=True),
              default=0.4, show_default=True,
              help="The correlation threshold when building graph from matrices.")
@click.pass_context
def correlations(ctx: click.core.Context, graph_path: Path, threshold: float):
    """Simulate correlations matrices from a spatio-temporal graph."""
    graph = load_spatio_temporal_graph(graph_path)
    simulator = CorrelationMatrixSequenceSimulator(graph, threshold=threshold)
    matrices = simulator.simulate()
    np.savez_compressed(ctx.obj, simulated=matrices)


if __name__ == '__main__':
    cli.add_command(plot)
    cli.add_command(simulate)
    cli()
