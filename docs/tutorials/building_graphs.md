# Building Spatio-Temporal Graphs

This tutorial walks through a complete example: preparing your input files, running the `build`
command, and inspecting the output archive.

## Prerequisites

- fSTG Toolkit installed (`pip install fSTG-Toolkit`)
- A NumPy file of correlation matrices (`.npz` or `.npy`)
- A CSV file describing brain areas and regions

## 1. Prepare the Areas Description

Create a CSV file with columns `Id_Area`, `Name_Area`, and `Name_Region`:

```csv
Id_Area,Name_Area,Name_Region
1,V1,Visual
2,V2,Visual
3,M1,Motor
4,M2,Motor
5,PFC,Prefrontal
```

Save it as `areas.csv`.

## 2. Prepare the Correlation Matrices

Your correlation matrices must be stored as a NumPy file. Each matrix sequence has shape
`(T, N, N)` where `T` is the number of time points and `N` matches the number of rows in
`areas.csv`.

If you have multiple subjects stored in a single `.npz` file, each key becomes a separate graph:

```python
import numpy as np

# Simulate two subjects, 10 time points, 5 brain areas
rng = np.random.default_rng(42)
matrices = {
    'subject1': rng.random((10, 5, 5)),
    'subject2': rng.random((10, 5, 5)),
}
np.savez_compressed('matrices.npz', **matrices)
```

## 3. Build the Graphs

Run the `build` command:

```shell
python -m fstg_toolkit graph build -o my_graphs.zip areas.csv matrices.npz
```

A progress bar shows the number of graphs being built (one per subject in the `.npz` file).
On completion, `my_graphs.zip` is created.

### Common options

| Flag | Default | Description |
|------|---------|-------------|
| `-o` | `output.zip` | Output archive path |
| `-t` | `0.4` | Correlation threshold |
| `--absolute-thresholding` | on | Threshold on absolute correlation values |
| `--no-raw` | off | Exclude raw matrices from the archive |
| `--max-cpus` | all-1 | Maximum number of parallel workers |

### Multiple input files

All `.npz` files can be passed together:

```shell
python -m fstg_toolkit graph build -o study.zip areas.csv batch1.npz batch2.npz batch3.npz
```

### Interactive selection

Use `--select` to choose which sequence to build when a file contains many subjects:

```shell
python -m fstg_toolkit graph build --select -o my_graph.zip areas.csv matrices.npz
```

## 4. Inspect the Output

The output `.zip` archive contains:
- `graphs/` — one JSON file per spatio-temporal graph
- `areas.csv` — the areas description
- `matrices/` — the raw correlation matrix files (unless `--no-raw` was passed)

You can load a graph programmatically:

```python
from fstg_toolkit import load_spatio_temporal_graph

graph = load_spatio_temporal_graph('my_graphs.zip', 'subject1')
print(f"Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")
```

## Next Steps

- [Computing Metrics](computing_metrics.md) — calculate spatial and temporal graph metrics
- [Running the Dashboard](running_dashboard.md) — visualise your graphs interactively
