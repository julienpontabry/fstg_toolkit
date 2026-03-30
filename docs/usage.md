# Usage

fSTG Toolkit provides a CLI with four command groups: `graph`, `plot`, `dashboard`, and `simulate`.

```shell
python -m fstg_toolkit --help
```

Use `--help` on any command or subcommand for detailed options:

```shell
python -m fstg_toolkit graph build --help
```

## Input File Formats

### Areas Description (CSV)

The areas/regions definition file is a CSV with at least three columns:

| Id_Area | Name_Area | Name_Region |
|---------|-----------|-------------|
| 1       | Area1     | Region1     |
| 2       | Area2     | Region1     |
| 3       | Area3     | Region2     |
| 4       | Area4     | Region3     |

```csv
Id_Area,Name_Area,Name_Region
1,Area1,Region1
2,Area2,Region1
3,Area3,Region2
4,Area4,Region3
```

The column names `Name_Area` and `Name_Region` are the defaults; they can be overridden with
`-acn` / `-rcn` flags on the `build` command.

### Correlation Matrices (NumPy)

Correlation matrices must be stored in a NumPy file (`.npz` or `.npy`):

- **`.npz`** — multiple named sequences, one matrix stack per key.
- **`.npy`** — a single matrix stack.

Each matrix stack has shape `(T, N, N)` where `T` is the number of time points and `N` is the
number of brain areas.

## Building Graphs

Build one or more spatio-temporal graphs from correlation matrices:

```shell
python -m fstg_toolkit graph build -o my_graph.zip areas.csv matrices.npz
```

Build from multiple files at once:

```shell
python -m fstg_toolkit graph build -o my_graphs.zip areas.csv m1.npz m2.npz m3.npz
```

Adjust the correlation threshold (default 0.4):

```shell
python -m fstg_toolkit graph build -t 0.5 -o my_graph.zip areas.csv matrices.npz
```

The output is a ZIP archive containing the graphs, the areas description, and the raw matrices.

## Calculating Metrics

Compute spatial and temporal graph metrics on a dataset archive:

```shell
python -m fstg_toolkit graph metrics my_graphs.zip
```

The metrics are written back into the archive.

## Frequent Pattern Mining

Discover frequent subgraph patterns (requires Docker and the `[frequent]` extra):

```shell
python -m fstg_toolkit graph frequent my_graphs.zip
```

The detected patterns are stored in the archive and can be explored in the dashboard.

## Visualising Results

### One-shot dashboard

Launch an interactive dashboard for a single dataset:

```shell
python -m fstg_toolkit dashboard show my_graphs.zip
```

### Persistent multi-dataset server

Serve a dashboard that accepts uploads and manages multiple datasets:

```shell
python -m fstg_toolkit dashboard serve <data_path> <upload_path>
```

## Plotting

Requires the `[plot]` extra.

```shell
# Multipartite layout (x = time, y = areas)
# NOTE: only for very small graphs
python -m fstg_toolkit plot my_graph.zip multipartite

# Spatial connectivity at time t=2
python -m fstg_toolkit plot my_graph.zip spatial -t 2

# Temporal connectivity
python -m fstg_toolkit plot my_graph.zip temporal

# Interactive dynamic plot
python -m fstg_toolkit plot my_graph.zip dynamic
```

## Factor and Subject Detection

When matrix names follow a structured naming convention, the dashboard automatically detects
factors and subjects. Name parts must be separated by underscores (`_`), slashes (`/`), or a
combination of both.

**Example names:**
- `control_time1_T21`
- `control/time2_T22`
- `group1_time2/T31`
- `group1_time1_T11`

In this case:
- **Subjects** (unique per name): `T21`, `T22`, `T31`, `T11`
- **Factors** (shared across names): `{control, group1}` and `{time1, time2}`
