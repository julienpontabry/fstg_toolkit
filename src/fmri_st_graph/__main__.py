import re
from pathlib import Path
from typing import Optional

import click
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from screeninfo import get_monitors

from fmri_st_graph import generate_pattern, SpatioTemporalGraphSimulator, CorrelationMatrixSequenceSimulator
from .factory import spatio_temporal_graph_from_corr_matrices
from .io import load_spatio_temporal_graph, save_spatio_temporal_graph, save_spatio_temporal_graphs
from .visualization import spatial_plot, temporal_plot, multipartite_plot, DynamicPlot
from .app.fstview import app


@click.group()
def cli():
    """Build, plot and simulate spatio-temporal graphs for fMRI data."""
    pass


## building ###################################################################

@cli.command()
@click.argument('correlation_matrices_path', type=click.Path(exists=True, path_type=Path))
@click.argument('areas_description_path', type=click.Path(exists=True, path_type=Path))
@click.option('-o', '--output', type=click.Path(writable=True, path_type=Path),
              default="output.zip", show_default="a graph archive named 'output.zip' in the current directory",
              help="Path where to write the built graph. If there is no '.zip' extension, it will be added.")
@click.option('-t', '--corr-threshold', type=click.FloatRange(min=0, max=1), default=0.4,
              show_default=True, help="The threshold of the correlations maps.")
@click.option('--absolute-thresholding/--no-absolute-thresholding', default=True,
              help="Flag to tell how to threshold the correlations (on absolute values or not).")
@click.option('-acn', '--areas-column-name', type=str, default='Name_Area',
              show_default=True, help="The name of the column of areas' names in the description file.")
@click.option('-rcn', '--regions-column-name', type=str, default='Name_Region',
              show_default=True, help="The name of the column of regions' names in the description file.")
@click.option('-s', '--select', is_flag=True, default=False,
              help="Select the graphs to build and save (only if there are multiple sets of correlation matrices).")
def build(correlation_matrices_path: Path, areas_description_path: Path, output: Path,
          corr_threshold: float, absolute_thresholding: bool, areas_column_name: str, regions_column_name: str,
          select: bool):
    """Build a spatio-temporal graph from correlation matrices.

    The spatio-temporal graph will be saved to OUTPUT, built from the correlation matrices in CORRELATION_MATRICES_PATH and the area descriptions in AREAS_DESCRIPTION_PATH.

    Accepted file formats for correlation matrices are numpy pickle files with extensions `.npz` or `.npy`.

    The CSV file for the description of areas and regions should have the columns: `Id_Area`, `Name_Area`, and `Name_Region`.

    If CORRELATION_MATRICES_PATH contains multiple sets of matrices, a list of available names will be displayed, and you will be prompted to choose one.
    """

    # read input matrices
    try:
        matrices = np.load(correlation_matrices_path)
    except Exception as ex:
        click.echo(f"Error while reading matrices: {ex}", err=True)
        exit(1)

    # select the matrices to process
    if isinstance(matrices, np.lib.npyio.NpzFile):
        if select:
            click.echo("The following sequences of matrices are available from the file:")
            click.echo(";\n".join(matrices.keys()) + ".")
            chosen = click.prompt("Which one to process?", default=next(iter(matrices.keys())))
            selected = [chosen]
        else:
            selected = matrices.keys()

        matrices = [(matrices[name], name) for name in selected]
    else:
        # TODO check that there is a single set of matrices in the file (shape should be (t, n, n))
        matrices = [(matrices, output)]

    # read input areas description
    try:
        areas = pd.read_csv(areas_description_path, index_col='Id_Area')
    except Exception as ex:
        click.echo(f"Error while reading areas description: {ex}", err=True)
        exit(1)

    # build the graphs
    graphs = {}
    with click.progressbar(matrices, label="Building ST graphs", show_pos=True,
                           item_show_func=lambda a: str(a[1]) if a is not None else None) as bar:
        for mat, name in bar:
            try:
                graphs[name] = spatio_temporal_graph_from_corr_matrices(
                    mat, areas, corr_thr=corr_threshold, abs_thr=absolute_thresholding,
                    area_col_name=areas_column_name, region_col_name=regions_column_name)
            except Exception as ex:
                click.echo(f"Error while processing {name}: {ex}", err=True)
                continue

    # save the graphs into a single zip file
    try:
        with click.progressbar(length=1, label="Saving ST graphs", show_eta=False, show_percent=False) as bar:
            save_spatio_temporal_graphs(graphs, output)
            bar.update(1)
    except OSError as ex:
        click.echo(f"Error while saving to {output}: {ex}", err=True)
        exit(1)


## plotting ###################################################################

def __figure_screen_setup(res_factor: float = 0.75, size_factor: float = 0.75):
    """Set up the figure dimensions and resolution based on the screen size.

    This function calculates the figure size and DPI (dots per inch) based on the screen dimensions
    and a resolution factor. It uses the first monitor detected by the `screeninfo` library.

    Parameters
    ----------
        res_factor: A (float) factor to adjust the resolution. Default is 0.65.
        size_factor: A (float) factor to adjust the size on the screen. Default is 0.75.

    Returns
    -------
    dict: A dictionary containing the figure size (`figsize`) and DPI (`dpi`).
    """
    monitor = next(iter(get_monitors()))
    screen_width = monitor.width_mm / 25.4
    screen_height = monitor.height_mm / 25.4
    dpi = max(monitor.width / screen_width, monitor.height / screen_height) * res_factor
    width = screen_width * size_factor / res_factor
    height = screen_height * size_factor / res_factor
    return dict(figsize=(width, height), dpi=dpi)


@click.group()
@click.argument('graph_path', type=click.Path(exists=True))
@click.pass_context
def plot(ctx: click.core.Context, graph_path: str):
    """Plot a spatio-temporal graph from an archive graph file.

    The file GRAPH_PATH must be an archive containing the spatio-temporal graph.
    """
    ctx.obj = load_spatio_temporal_graph(graph_path)


@plot.command()
@click.pass_context
def multipartite(ctx: click.core.Context):
    """Plot as a multipartite graph.

    The x-axis represents time, with nodes at each time point aligned vertically.

    Note: This plot requires significant memory and may not be suitable for large graphs.
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
        fig, axe = plt.subplots(layout='constrained', **__figure_screen_setup())
        multipartite_plot(ctx.obj, ax=axe)
        plt.show()


@plot.command()
@click.option('-t', '--time', type=click.IntRange(0),
              default=0, show_default=True,
              help="The time index of the spatial subgraph to show.")
@click.pass_context
def spatial(ctx: click.core.Context, time: int):
    """Plot as a spatial connectivity graph.

    Displays only the nodes and spatial edges at a specified time.
    """
    max_time = ctx.obj.graph['max_time']
    if time > max_time:
        click.echo(f"The graph as a maximum time of {max_time} and "
                   f"requested time is greater ({time}>{max_time})!")
    else:
        time = min(time, max_time)
        fig, axe = plt.subplots(layout='constrained', **__figure_screen_setup())
        spatial_plot(ctx.obj, time, ax=axe)
        plt.show()


@plot.command()
@click.pass_context
def temporal(ctx: click.core.Context):
    """Plot as a temporal connectivity graph.

    Displays all nodes and only the temporal edges.
    """
    fig, axe = plt.subplots(layout='constrained', **__figure_screen_setup())
    temporal_plot(ctx.obj, ax=axe)
    plt.show()


@plot.command()
@click.pass_context
def dynamic(ctx: click.core.Context):
    """Display an interactive dynamic graph.

    Shows both spatial and temporal graphs with interactivity.
    """
    DynamicPlot(ctx.obj).plot(__figure_screen_setup(size_factor=1))
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
    elem_desc = r'\s*(?P<range>\d+(:\d+)?),(?P<id>\d+),(?P<strength>-?\d*.?\d+)'

    def _convert_from_match(self, match: re.Match[str]) -> tuple[any, ...]:
        tmp = match.group('range').split(':')
        areas = int(tmp[0]) if len(tmp) == 1 else (int(tmp[0]), int(tmp[1]))
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
    """Generate a spatio-temporal graph pattern from a description.

    The input strings for networks, spatial edges, and temporal edges must follow specific formats.

    NETWORKS:

        The syntax for a single network is `area_range,region,internal_strength`, where `area_range` is either a single area ID or a range between IDs separated by a colon. Multiple networks at a given time are concatenated with spaces. Networks of different time instants are separated by a `/` symbol. The entire description must be surrounded by quotes.

    SPATIAL_EDGES:

        The syntax for a single spatial edge is `network1_id,network2_id,correlation`. Multiple descriptions are concatenated between quotes and separated by spaces.

    TEMPORAL_EDGES:

        The syntax for a single temporal edge is `network_id_range,network_id_range`, where `network_id_range` can be either a single network ID or multiple IDs separated by a `-` character. The type of edge is automatically inferred. Multiple descriptions are concatenated between quotes and separated by spaces.
    """
    if spatial_edges is None:
        spatial_edges = []

    if temporal_edges is None:
        temporal_edges = []

    pat = generate_pattern(networks_list=networks,
                           spatial_edges=spatial_edges,
                           temporal_edges=temporal_edges)
    save_spatio_temporal_graph(pat, ctx.obj)


@simulate.command()
@click.argument('patterns', nargs=-1, type=click.Path(exists=True))
@click.argument('sequence_description', type=GRAPH_SEQUENCE_DESCRIPTION)
@click.pass_context
def sequence(ctx: click.core.Context, patterns: tuple[Path], sequence_description: list[str | int]):
    """Generate a spatio-temporal graph from a sequence of patterns.

    PATTERNS are the paths to the pattern files used to generate the graph.

    SEQUENCE_DESCRIPTION is a space-separated list of elements, where each element is either a pattern (p<n>, where n is the order of the pattern) or a number (d) indicating d steady states.
    """
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


## showing #################################################################

@cli.command()
@click.option('--debug', is_flag=True, default=False,
              help="Run the dashboard in debug mode.")
@click.option('-p-', '--port', type=int, default=8050, show_default=True,
              help="Port to run the dashboard on.")
def show(debug: bool, port: int):
    """Show a dashboard for visualizing spatio-temporal graphs."""
    app.run(debug=debug, port=port)


if __name__ == '__main__':
    cli.add_command(plot)
    cli.add_command(simulate)
    cli()
