import click
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from .factory import spatio_temporal_graph_from_corr_matrices
from .io import load_spatio_temporal_graph, save_spatio_temporal_graph
from .visualization import spatial_plot, temporal_plot, multipartite_plot


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
        chosen = click.prompt("Which one to process?", default=list(matrices.keys())[0])
        matrices = matrices[chosen]

    areas = pd.read_csv(areas_description_path, index_col='Id_Area')
    graph = spatio_temporal_graph_from_corr_matrices(matrices, areas)
    save_spatio_temporal_graph(graph, output_graph)


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
    temporal_plot(ctx.obj)
    plt.show()


if __name__ == '__main__':
    cli.add_command(plot)
    cli()
