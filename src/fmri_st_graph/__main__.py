from pathlib import Path

import click
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from .factory import spatio_temporal_graph_from_corr_matrices
from .io import load_spatio_temporal_graph, save_spatio_temporal_graph
from .visualization import spatial_plot, temporal_plot, multipartite_plot, dynamic_plot


@click.group()
def cli():
    pass


@cli.command()
@click.argument('correlation_matrices_path', type=click.Path(exists=True))
@click.argument('areas_description_path', type=click.Path(exists=True))
@click.option('-o', '--output_graph', type=click.Path(writable=True), default="st_graph.zip",
              help="Path where to write the built graph.")
def build(correlation_matrices_path: str, areas_description_path: str, output_graph: str):
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


@click.group()
@click.argument('graph_path', type=click.Path(exists=True))
@click.pass_context
def plot(ctx: click.core.Context, graph_path: str):
    ctx.obj = load_spatio_temporal_graph(graph_path)


@plot.command()
@click.pass_context
def multipartite(ctx: click.core.Context):
    multipartite_plot(ctx.obj)
    plt.show()


@plot.command()
@click.option('-t', '--time', type=int, default=0,
              help="The time index of the spatial subgraph to show.")
@click.pass_context
def spatial(ctx: click.core.Context, time: int):
    spatial_plot(ctx.obj, time)
    plt.show()


@plot.command()
@click.pass_context
def temporal(ctx: click.core.Context):
    fig, axe = plt.subplots(layout='constrained')
    temporal_plot(ctx.obj, ax=axe)
    plt.show()


@plot.command()
@click.option('-s', '--size', type=click.FloatRange(0), default=70,
              help="The size of the plotting window (in centimeter).")
@click.pass_context
def dynamic(ctx: click.core.Context, size: float):
    dynamic_plot(ctx.obj, size)
    plt.show()


if __name__ == '__main__':
    cli.add_command(plot)
    cli()
