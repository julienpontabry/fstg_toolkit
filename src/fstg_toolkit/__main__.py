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

__help_epilog = []

import logging.config
import multiprocessing
import re
import tempfile
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib.resources import files
from pathlib import Path
from typing import Optional, Tuple, Any, List, Generator, Callable

import numpy as np
import pandas as pd
import rich_click as click
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, TextColumn, MofNCompleteColumn, BarColumn, TimeRemainingColumn, TaskProgressColumn, \
    SpinnerColumn
from rich.prompt import Prompt
from screeninfo import get_monitors

try:
    from matplotlib import pyplot as plt
except ImportError:
    plt = None
    __help_epilog.append(f"⚠️  Install '{__package__}[plot]' to unlock the plotting commands.")

from fstg_toolkit import generate_pattern, SpatioTemporalGraphSimulator, CorrelationMatrixSequenceSimulator
from .factory import spatio_temporal_graph_from_corr_matrices
from .graph import SpatioTemporalGraph
from .io import save_spatio_temporal_graph, DataSaver, DataLoader, save_metrics
from .app.core.io import GraphsDataset
from .metrics import calculate_spatial_metrics, calculate_temporal_metrics, gather_metrics

try:
    from .visualization import spatial_plot, temporal_plot, multipartite_plot, DynamicPlot
except ImportError:
    spatial_plot = temporal_plot = multipartite_plot = DynamicPlot = None

try:
    from .app.fstg_view import app
    from .app.core.config import config
    from .app.core.datafilesdb import get_data_file_db, MemoryDataFilesDB, SQLiteDataFilesDB
except ImportError as e:
    app = config = get_data_file_db = MemoryDataFilesDB = SQLiteDataFilesDB = None
    __help_epilog.append(f"⚠️  Install '{__package__}[dashboard]' to unlock the dashboard commands.")

try:
    from .frequent import SPMinerService
except ImportError as e:
    SPMinerService = None
    __help_epilog.append(f"⚠️  Install '{__package__}[frequent]' to unlock the frequent patterns analysis command.")


console = Console()
error_console = Console(stderr=True, style="bold red")


class __OrderedGroup(click.RichGroup):
    def list_commands(self, ctx):
        return list(self.commands)


@click.group(context_settings={'help_option_names': ['-h', '--help']},
             cls=__OrderedGroup, epilog="\n\n".join(__help_epilog))
@click.version_option(None, '--version', '-v', package_name=__package__, prog_name=__package__)
def cli():
    """Build, plot and simulate spatio-temporal graphs for fMRI data."""
    pass


## data utils #################################################################

def __load_graph(filepath: Path) -> SpatioTemporalGraph:
    loader = DataLoader(Path(filepath))
    filenames = loader.lazy_load_graphs()

    if (n := len(filenames)) > 0:
        if n == 1:
            chosen = filenames[0]
        else:
            console.print("The following graphs are available from the input dataset:")
            console.print(Panel.fit("\n".join(filenames)))
            chosen = Prompt.ask("Which graph to load?", default=filenames[0])
    else:
        error_console.print("No graph found in data file.")
        error_console.print_exception()
        exit(1)

    areas = loader.load_areas()
    return loader.load_graph(areas, chosen)


## building and computing #####################################################


@click.group()
def graph():
    """Build, calculate metrics and simulate graphs."""
    pass


def __read_load_np(path: Path) -> List[Tuple[str, np.ndarray]]:
    """
    Reads a numpy file (.npz or .npy) and returns a list of tuples containing the matrices and their names.

    Parameters
    ----------
    path : Path
        Path to the numpy file to load.

    Returns
    -------
    List[Tuple[str, np.ndarray]]
        List of tuples (matrix, name) extracted from the file.
    """
    red = np.load(path)

    if isinstance(red, np.lib.npyio.NpzFile):
        return [(name, matrices) for name, matrices in red.items()]
    else:
        return [(path.name, red)]


def _build_graph(name: str, matrix: np.ndarray, areas: pd.DataFrame, corr_threshold: float,
                 absolute_thresholding: bool, areas_column_name: str,
                 regions_column_name: str) -> Tuple[str, Optional[SpatioTemporalGraph]]:
    try:
        return name, spatio_temporal_graph_from_corr_matrices(
            matrix, areas, corr_thr=corr_threshold, abs_thr=absolute_thresholding,
            area_col_name=areas_column_name, region_col_name=regions_column_name)
    except Exception as ex:
        error_console.print(f"Error while processing {name}: {ex}")
        error_console.print_exception()
        return name, None

from contextlib import contextmanager

@contextmanager
def _progress_factory(description: str, message_after: Optional[str|Callable[[], str]] = None,
                      steps: bool = False, transient: bool = False,
                      spinner: bool = True) -> Generator[Progress, None, None]:
    columns = [SpinnerColumn()] if spinner else []
    columns += [
        TextColumn(description),
        BarColumn(),
        MofNCompleteColumn() if steps else TaskProgressColumn(),
        TimeRemainingColumn(compact=True, elapsed_when_finished=True),
        TextColumn("{task.description}")
    ]

    try:
        with Progress(*columns, transient=transient) as bar:
            yield bar
    finally:
        if message_after:
            console.print(message_after if isinstance(message_after, str) else message_after())


@graph.command()
@click.argument('areas_description_path', type=click.Path(exists=True, path_type=Path))
@click.argument('correlation_matrices_path', type=click.Path(exists=True, path_type=Path), nargs=-1)
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
@click.option('--select', is_flag=True, default=False,
              help="Select the graphs to build and save using an input prompt "
                   "(only if there are multiple sets of correlation matrices).")
@click.option('--no-raw', is_flag=True, default=False,
              help="Do not save the raw data along with the graphs.")
@click.option('--max-cpus', type=click.IntRange(1, multiprocessing.cpu_count()-1), default=multiprocessing.cpu_count()-1,
              help="Set the number of CPUs to use for the processing.")
def build(areas_description_path: Path, correlation_matrices_path: Tuple[Path], output: Path,
          corr_threshold: float, absolute_thresholding: bool, areas_column_name: str, regions_column_name: str,
          select: bool, no_raw: bool, max_cpus: int):
    """Build spatio-temporal graphs from sequences of correlation matrices.

    The spatio-temporal graphs will be saved to OUTPUT, built from the correlation matrices in CORRELATION_MATRICES_PATH and the area descriptions in AREAS_DESCRIPTION_PATH.

    Accepted file formats for correlation matrices are numpy pickle files with extensions `.npz` or `.npy`.

    The CSV file for the description of areas and regions should have the columns: `Id_Area`, `Name_Area`, and `Name_Region`.
    """

    # prepare the data saver
    saver = DataSaver()

    # read input matrices
    try:
        with console.status(f"Reading {len(correlation_matrices_path)} files..."):
            matrices = {}
            for cm in correlation_matrices_path:
                for name, matrix in __read_load_np(cm):
                    matrices[name] = matrix
                console.print(f"- red '{cm.name}'")
        console.print(f"Red {len(correlation_matrices_path)} files.")
    except Exception as ex:
        error_console.print(f"Error while reading matrices: {ex}")
        error_console.print_exception()
        exit(1)

    # select the matrices to process
    keys = list(matrices.keys())

    if select:
        console.print("The following sequences of matrices are available from the file:")
        console.print(Panel.fit("\n".join(keys)))
        chosen = Prompt.ask("Which one to process?", default=keys[0])
        selected = [chosen]
    else:
        selected = keys

    matrices = {name: matrix for name, matrix in matrices.items() if name in selected}

    if not no_raw:
        saver.add(matrices)

    # read input areas description
    try:
        areas = pd.read_csv(areas_description_path, index_col='Id_Area')
        saver.add(areas)
    except Exception as ex:
        error_console.print(f"Error while reading areas description: {ex}")
        error_console.print_exception()
        exit(1)

    # build the graphs
    graphs = {}

    with _progress_factory("Building ST graphs...", lambda: f"Built {len(graphs)} ST graphs.",
                           steps=True, transient=True) as bar:
        task = bar.add_task("", total=len(matrices))

        futures = []
        num_workers = min(len(matrices), max_cpus)

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for name, matrix in matrices.items():
                future = executor.submit(_build_graph, name, matrix, areas,
                                         corr_threshold, absolute_thresholding,
                                         areas_column_name, regions_column_name)
                futures.append(future)

            for future in as_completed(futures):
                name, graph = future.result()
                if graph:
                    graphs[name] = graph
                bar.update(task, advance=1, description=name)

    saver.add(graphs)

    # save the graphs into a single zip file
    try:
        with console.status("Saving dataset..."):
            saver.save(output)
        console.print(f"Dataset saved to '{output}'.")
    except OSError as ex:
        error_console.print(f"Error while saving to {output}: {ex}")
        error_console.print_exception()
        exit(1)


@graph.command()
@click.argument('dataset_path', type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option('--max-cpus', type=click.IntRange(1, multiprocessing.cpu_count()-1), default=multiprocessing.cpu_count()-1,
              help="Set the number of CPUs to use for the processing.")
def metrics(dataset_path: Path, max_cpus: int):
    """Calculate metrics on spatio-temporal graphs.

    DATASET_PATH is the path to spatio-temporal graphs built with the command 'build'.
    """
    dataset = GraphsDataset.from_filepath(dataset_path)

    with _progress_factory("Calculating local metrics...",
                           lambda: f"Local metrics calculated on {len(spatial_df)} spatial graphs.",
                           steps=True, transient=True) as bar:
        task = bar.add_task("", total=len(dataset.subjects))
        spatial_df = gather_metrics(dataset, dataset.subjects.index, calculate_spatial_metrics,
                                    callback=lambda s: bar.update(task, advance=1), max_cpus=max_cpus)

    with _progress_factory("Calculating global metrics...",
                           lambda: f"Global metrics calculated on {len(temporal_df)} ST graphs.",
                           steps=True, transient=True) as bar:
        task = bar.add_task("", total=len(dataset.subjects))
        temporal_df = gather_metrics(dataset, dataset.subjects.index, calculate_temporal_metrics,
                                     callback=lambda s: bar.update(task, advance=1), max_cpus=max_cpus)

    # TODO modify the data saver to accepts those files
    # save the metrics into the dataset
    with zipfile.ZipFile(dataset_path, 'a') as zfp:
        try:
            # TODO handle already present files
            with console.status("Saving local metrics..."):
                with zfp.open('metrics_local.csv', 'w') as fp:
                    save_metrics(fp, spatial_df)
            console.print(f"Local metrics saved in '{dataset_path}'.")
        except OSError as ex:
            error_console.print(f"Error while saving local metrics to {dataset_path}: {ex}")
            error_console.print_exception()

        try:
            # TODO handle already present files
            with console.status("Saving global metrics..."):
                with zfp.open('metrics_global.csv', 'w') as fp:
                    save_metrics(fp, temporal_df)
            console.print(f"Global metrics saved in '{dataset_path}'.")
        except OSError as ex:
            error_console.print(f"Error while saving global metrics to {dataset_path}: {ex}")
            error_console.print_exception()


@click.command()
@click.argument('dataset_path', type=click.Path(exists=True, dir_okay=False, path_type=Path))
def frequent(dataset_path: Path):
    """Perform a frequent patterns analysis.

    It relies on the SPMiner package. Frequent patterns will be written into the given dataset.

    Note that loading the SPMiner docker service for the first time can take a while, as its
    docker image is built at the first use. Later call will be much faster.

    DATASET_PATH is the path to spatio-temporal graphs built with the command 'build'.
    """

    # load SPMiner service
    service = SPMinerService()

    with console.status("Loading SPMiner service..."):
        service.prepare()
    console.print("SPMiner service loaded.")

    # process graphs in dataset
    with tempfile.TemporaryDirectory() as input_dir:
        input_dir = Path(input_dir)

        # extract dataset's graphs in the input temporary directory
        with _progress_factory("Preparing dataset...", "Dataset prepared.", steps=True, transient=True) as bar:
            task = bar.add_task("",  total=None)

            # TODO parallelize the graph extraction
            # FIXME better handle IO for datasets
            with zipfile.ZipFile(dataset_path, 'r') as zfp:
                files = [file for file in zfp.namelist() if Path(file).suffix == '.json' and Path(file).stem != 'motifs_enriched_t']
                for file in files:
                    zfp.extract(file, input_dir)
                    bar.update(task, advance=1, total=len(files))

        with tempfile.TemporaryDirectory() as output_dir:
            output_dir = Path(output_dir)

            # run service and gather output files
            with _progress_factory("Running SPMiner...", "SPMiner analysis completed.", steps=True, transient=True) as bar:
                task = bar.add_task("", total=None)
                for completed, total in service.run(input_dir, output_dir):
                    bar.update(task, completed=completed, total=total)

            # save found frequent patterns into the dataset
            # TODO parallelize the patterns inclusion
            # FIXME better handle IO for datasets
            with _progress_factory("Saving frequent patterns...", f"Frequent patterns saved to dataset '{dataset_path}'.",
                                   steps=True, transient=True) as bar:
                task = bar.add_task("", total=None)

                with zipfile.ZipFile(dataset_path, 'a') as zfp:
                    files = list(output_dir.rglob('*.json'))
                    for file in files:
                        zfp.write(str(file), str(file.relative_to(output_dir)))
                        bar.update(task, advance=1, total=len(files))

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
    return {'figsize': (width, height), 'dpi': dpi}


@click.group()
@click.argument('graph_path', type=click.Path(exists=True, path_type=Path))
@click.pass_context
def plot(ctx: click.core.Context, graph_path: Path):
    """Plot a spatio-temporal graph from an archive graph file.

    The file GRAPH_PATH must be an archive containing the spatio-temporal graph.
    """
    ctx.obj = __load_graph(graph_path)


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
        console.print(f"[yellow]⚠️  There are {n} nodes in this graph; the multipartite plot has "
                       "not been optimized for big graphs![/yellow]")
        answer = Prompt.ask("Do you really want to continue?", choices=["yes", "no"], default="no")
        go_on = answer == 'yes'

    if go_on:
        _, axe = plt.subplots(layout='constrained', **__figure_screen_setup())
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
        error_console.print(f"The graph as a maximum time of {max_time} and "
                            f"requested time is greater ({time}>{max_time})!")
    else:
        time = min(time, max_time)
        _, axe = plt.subplots(layout='constrained', **__figure_screen_setup())
        spatial_plot(ctx.obj, time, ax=axe)
        plt.show()


@plot.command()
@click.pass_context
def temporal(ctx: click.core.Context):
    """Plot as a temporal connectivity graph.

    Displays all nodes and only the temporal edges.
    """
    _, axe = plt.subplots(layout='constrained', **__figure_screen_setup())
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

    def _convert_from_match(self, match: re.Match[str]) -> Tuple[Any,...]:
        pass

    def convert(self, value: Any, param: Optional[click.Parameter], ctx: Optional[click.Context]) -> Any:
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

    def _convert_from_match(self, match: re.Match[str]) -> Tuple[Any, ...]:
        tmp = match.group('range').split(':')
        areas = int(tmp[0]) if len(tmp) == 1 else (int(tmp[0]), int(tmp[1]))
        region = int(match.group('id'))
        internal_strength = float(match.group('strength'))
        return areas, region, internal_strength


class NetworksDescription(click.ParamType):
    _delegate = _NetworkDescription()

    def convert(self, value: Any, param: Optional[click.Parameter], ctx: Optional[click.Context]) -> Any:
        if not isinstance(value, str):
            self.fail(f"{value!r} is not a valid networks description!")
        return [self._delegate.convert(network_desc, param, ctx)
                for network_desc in value.split('/')]

NETWORKS_DESCRIPTION = NetworksDescription()


class SpatialEdgesDescription(GraphElementsDescription):#click.ParamType):
    elem_desc = r'(?P<n1>\d+),(?P<n2>\d+),(?P<corr>-?\d*.\d+)'

    def _convert_from_match(self, match: re.Match[str]) -> Tuple[Any,...]:
        node1 = int(match.group('n1'))
        node2 = int(match.group('n2'))
        correlation = float(match.group('corr'))
        return node1, node2, correlation

SPATIAL_EDGES_DESCRIPTION = SpatialEdgesDescription()


class TemporalEdgesDescription(GraphElementsDescription):
    main_desc = r'[-,.\w]+'
    elem_desc = r'(?P<n1>\d+(-\d+)?),(?P<n2>\d+(-\d+)?)'

    @staticmethod
    def __build_range(s: str) -> Tuple[int, int] | int:
        if '-' in s:
            tmp = s.split('-')
            return int(tmp[0]), int(tmp[1])
        else:
            return int(s)

    def _convert_from_match(self, match: re.Match[str]) -> Tuple[Any,...]:
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
    def convert(self, value: Any, param: Optional[click.Parameter], ctx: Optional[click.Context]) -> Any:
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


@graph.group()
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
def pattern(ctx: click.core.Context, networks: List[List[Tuple[Tuple[int, int], int, float]]],
            spatial_edges: List[Tuple[int, int, float]] | None,
            temporal_edges: List[Tuple[int, int, str]] | None):
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
def sequence(ctx: click.core.Context, patterns: Tuple[Path], sequence_description: List[str | int]):
    """Generate a spatio-temporal graph from a sequence of patterns.

    PATTERNS are the paths to the pattern files used to generate the graph.

    SEQUENCE_DESCRIPTION is a space-separated list of elements, where each element is either a pattern (p<n>, where n is the order of the pattern) or a number (d) indicating d steady states.
    """
    pattern_graphs = {f'p{i+1}': __load_graph(filepath)
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
    graph = __load_graph(graph_path)
    simulator = CorrelationMatrixSequenceSimulator(graph, threshold=threshold)
    matrices = simulator.simulate()
    np.savez_compressed(ctx.obj, simulated=matrices)


## showing #################################################################

@click.group()
def dashboard():
    """Show dashboards for visualizing spatio-temporal graphs."""

@dashboard.command()
@click.argument('graphs-data',
                type=click.Path(exists=True, dir_okay=False, readable=True, executable=False, path_type=Path))
@click.option('--debug', is_flag=True, default=False,
              help="Run the dashboard in debug mode.")
@click.option('-p', '--port', type=int, default=8050, show_default=True,
              help="Port to run the dashboard on.")
@click.option('--no-browser', is_flag=True, default=False,
              help="Start without opening the app in the default browser.")
def show(graphs_data: Path, debug: bool, port: int, no_browser: bool):
    """Show a dashboard for visualizing spatio-temporal graphs."""

    db = get_data_file_db(requested_type=MemoryDataFilesDB, debug=debug)
    token = db.add(graphs_data)

    if not no_browser:
        click.launch(f'http://127.0.0.1:8050/dashboard/{token}')
    else:
        console.print(f"Dashboard for file {graphs_data} is at URL http://127.0.0.1:8050/dashboard/{token}")

    app.run(debug=debug, port=port)


@dashboard.command()
@click.argument('data_path', type=click.Path(exists=True, file_okay=False, readable=True, writable=True, path_type=Path))
@click.argument('upload_path', type=click.Path(exists=True, file_okay=False, readable=True, writable=True, path_type=Path))
@click.option('--debug', is_flag=True, default=False,
              help="Run the dashboard in debug mode.")
@click.option('-p', '--port', type=int, default=8050, show_default=True,
              help="Port to run the dashboard service on.")
@click.option('-d', '--db-path', type=click.Path(dir_okay=False, path_type=Path),
              default=Path.cwd() / 'data_files.db', show_default="a 'data_files.db' file in the current directory",
              help="Path to the database file to use for storing data files information.")
def serve(data_path: Path, upload_path: Path, debug: bool, port: int, db_path: Path):
    """Serve a dashboard for visualizing spatio-temporal graphs from a data directory."""

    # set up the configuration
    config.data_path = data_path
    config.upload_path = upload_path
    config.db_path = db_path

    # set up the data file database
    get_data_file_db(requested_type=SQLiteDataFilesDB, db_path=db_path, debug=debug)

    # set up and run the dash app
    console.print(f"Dashboard serving data from {data_path} is at URL http://127.0.0.1:8050")
    app.run(debug=debug, port=port)


if __name__ == '__main__':
    # setup logging
    logging_config_path = files(__package__).joinpath('logging.yml')
    with logging_config_path.open() as f:
        logging_config = yaml.safe_load(f.read())
        logging.config.dictConfig(logging_config)

    # setup CLI
    cli.add_command(graph)

    if plt is not None:
        cli.add_command(plot)

    if app is not None:
        cli.add_command(dashboard)

    if SPMinerService is not None:
        graph.add_command(frequent)

    cli()
