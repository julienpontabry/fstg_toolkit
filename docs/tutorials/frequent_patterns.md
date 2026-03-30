# Frequent Subgraph Pattern Mining

fSTG Toolkit integrates {ref}`SPMiner <spminer-ref>` to detect frequently recurring subgraph patterns across all spatio-temporal graphs in a dataset.

## Prerequisites

1. **Docker** must be installed and running on your machine. See
   [Docker installation docs](https://docs.docker.com/get-docker/).

2. Install the `[frequent]` extra:

   ```shell
   pip install "fSTG-Toolkit[frequent]"
   ```

## Running Pattern Mining

```shell
python -m fstg_toolkit graph frequent my_graphs.zip
```

On the **first run**, the SPMiner Docker image is built automatically. This may take several
minutes. Subsequent runs use the cached image and are much faster.

The command:
1. Extracts the graphs from the archive into a temporary directory.
2. Starts the SPMiner Docker service.
3. Runs the subgraph mining algorithm.
4. Saves the detected frequent patterns back into the archive.

## Viewing the Results

After mining completes, open the dashboard to explore the detected patterns:

```shell
python -m fstg_toolkit dashboard show my_graphs.zip
```

Navigate to the **Patterns** tab. Each detected pattern is shown as a small subgraph with its
support (the fraction of subjects in which the pattern appears).

## Programmatic Access

Frequent patterns can be loaded directly from the archive:

```python
from fstg_toolkit.io import DataLoader

loader = DataLoader('my_graphs.zip')
patterns = loader.load_frequent_patterns()

for name, pattern in patterns.items():
    print(f"{name}: {pattern.number_of_nodes()} nodes, support={pattern.graph.get('support')}")
```

## Troubleshooting

**Docker not found**
: Ensure Docker is installed and the `docker` CLI is on your `PATH`. Run `docker info` to
  verify the daemon is running. Also, you must be able to run docker in [rootless mode](https://docs.docker.com/engine/security/rootless/).

**First docker image build takes a while**
: The SPMiner image is built from source on first use. This is expected; subsequent calls reuse
  the cached image.

**Mining frequent patterns may take a long time**
: Depending on the size of your graphs and the number of timepoints, mining the frequent may take quite a long time. A progress bar help you to follow the progress.

**No patterns found**
: SPMiner may return empty results if the dataset is small or the graphs are too dissimilar.
  Try increasing the dataset size or adjusting the correlation threshold when building graphs.

## References

- (spminer-ref)=
  Zeghina et al. (2023) *Multi-SPMiner: A Deep Learning Framework for Multi-Graph Frequent Pattern Mining with Application to spatiotemporal Graphs*, Procedia Computer Science, vol. 225, pp. 1094-1103, ISSN 1877-0509, DOI: 10.1016/j.procs.2023.10.097.

## Next Steps

- [Building Graphs](building_graphs.md)
- [Running the Dashboard](running_dashboard.md)
- [API Reference: frequent](../api/frequent.rst)
