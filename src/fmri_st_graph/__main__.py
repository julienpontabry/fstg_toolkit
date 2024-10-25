import click
from matplotlib import pyplot as plt

from .io import load_spatio_temporal_graph
from .visualization import spatial_plot, temporal_plot, multipartite_plot


@click.group()
@click.argument('path', type=click.Path(exists=True))
@click.pass_context
def plot(ctx: click.core.Context, path: str):
    ctx.obj = load_spatio_temporal_graph(path)


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
    plot()
