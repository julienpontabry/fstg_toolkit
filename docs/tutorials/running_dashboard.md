# Running the Dashboard

The fSTG Toolkit dashboard provides an interactive web interface for exploring spatio-temporal
graphs, metrics, raw correlation matrices, and frequent patterns.

## Prerequisites

Install the `[dashboard]` extra:

```shell
pip install "fSTG-Toolkit[dashboard]"
```

## One-Shot Mode: Exploring a Single Dataset

The `show` command starts a local server and opens your default browser automatically:

```shell
python -m fstg_toolkit dashboard show my_graphs.zip
```

The dashboard will be available at `http://127.0.0.1:8050/dashboard/<token>`.

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--port` / `-p` | `8050` | Port to listen on |
| `--no-browser` | off | Start the server without opening a browser tab |
| `--debug` | off | Enable Dash debug mode with hot-reload |

```shell
python -m fstg_toolkit dashboard show --port 8080 --no-browser my_graphs.zip
```

## Persistent Mode: Multi-Dataset Server

Use the `serve` command to run a persistent server that supports multiple datasets and file
uploads:

```shell
python -m fstg_toolkit dashboard serve /path/to/data /path/to/uploads
```

Both paths must be existing, writable directories.

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--port` / `-p` | `8050` | Port to listen on |
| `--db-path` / `-d` | `./data_files.db` | SQLite database for tracking uploaded datasets |
| `--token-size` / `-t` | `3` | Token length for dataset URLs (increase for many simultaneous users) |
| `--debug` | off | Enable Dash debug mode |

## Dashboard Pages

### Home

Overview of the toolkit and links to loaded datasets.

### Dataset List

Lists all datasets currently loaded in the server. Click a dataset to open its dashboard.

### Dashboard

The main interactive view for a single dataset. Contains:

- **Data tab** — subjects and factors detected from matrix names; filter selector
- **Matrices tab** — heatmap visualisation of the raw correlation matrices
- **Graph tab** — spatio-temporal graph visualisation per subject
- **Metrics tab** — interactive charts of spatial and temporal metrics
- **Patterns tab** — frequent subgraph patterns (if pattern mining has been run)

### Submit

Upload new `.zip` datasets to the server (persistent mode only).

## Factor and Subject Filtering

If your matrix names follow the `factor_factor_subject` naming convention, the dashboard
automatically detects factors and presents them as filter dropdowns in the Data tab.
See [Usage: Factor and Subject Detection](../usage.md#factor-and-subject-detection) for details.

![Dashboard screenshot](../_static/images/illustration_web-viewer.png)

## Next Steps

- [Frequent Pattern Mining](frequent_patterns.md) — discover recurring connectivity patterns
- [API Reference: app](../api/app/index.rst) — dashboard internals
