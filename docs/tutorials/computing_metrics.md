# Computing Graph Metrics

After building your spatio-temporal graphs, you can compute a set of spatial and temporal metrics
that characterise the connectivity patterns in your data.

## Prerequisites

A dataset archive built with `graph build` (see [Building Graphs](building_graphs.md)).

## Running the Metrics Command

```shell
python -m fstg_toolkit graph metrics my_graphs.zip
```

The command reads all graphs from the archive, computes metrics in parallel, and writes the results
back into the same archive. No separate output file is created.

### Parallel computation

By default, the command uses all available CPU cores minus one. You can limit this:

```shell
python -m fstg_toolkit graph metrics --max-cpus 4 my_graphs.zip
```

## Available Metrics

### Spatial Metrics (per node, per time step)

Spatial metrics are computed on the spatial subgraph at each time point:

| Metric | Description |
|--------|-------------|
| Degree | Number of connections of each brain area |
| Betweenness centrality | How often a node lies on the shortest path between two others |
| Clustering coefficient | Local clique density around each node |
| PageRank | Relative importance of nodes via random walk |

### Temporal Metrics (per graph)

Temporal metrics characterise the structure and evolution of the full spatio-temporal graph:

| Metric | Description |
|--------|-------------|
| Number of nodes | Total number of nodes across all time steps |
| Number of temporal edges | Number of transitions between time steps |
| Graph density | Ratio of existing edges to possible edges |
| Temporal path length | Average shortest path length along temporal edges |

## Accessing Metrics Programmatically

After running the command, metrics are stored in the archive and can be loaded:

```python
from fstg_toolkit.io import DataLoader

loader = DataLoader('my_graphs.zip')
metrics = loader.load_metrics()

# metrics['local']  → pandas DataFrame, spatial metrics per node/time
# metrics['global'] → pandas DataFrame, temporal metrics per graph
print(metrics['local'].head())
print(metrics['global'].head())
```

## Next Steps

- [Running the Dashboard](running_dashboard.md) — explore your metrics visually in the dashboard
- [API Reference: metrics](../api/metrics.rst) — full API documentation for `calculate_spatial_metrics` and `calculate_temporal_metrics`
