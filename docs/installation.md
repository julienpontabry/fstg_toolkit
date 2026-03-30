# Installation

## Requirements

fSTG Toolkit requires **Python 3.12 or 3.13**. Earlier versions are not supported.

## Install from PyPI

The recommended way to get started is to install the package from PyPI:

```shell
pip install fSTG-Toolkit
```

This installs the core library and CLI. Optional feature sets can be installed as extras:

```shell
pip install "fSTG-Toolkit[dashboard]"               # interactive web dashboard
pip install "fSTG-Toolkit[plot]"                    # matplotlib-based plots
pip install "fSTG-Toolkit[frequent]"                # frequent pattern mining (requires Docker)
pip install "fSTG-Toolkit[dashboard,plot,frequent]" # all features
```

## Install from Source

Clone the repository and create a dedicated environment using conda:

```shell
conda env create -n <env_name> -f environment.yml
conda activate <env_name>
```

Then install the package with Poetry:

```shell
poetry install
```

To include optional feature sets:

```shell
poetry install --extras dashboard   # web dashboard
poetry install --extras plot        # matplotlib plots
poetry install --extras frequent    # frequent pattern mining
poetry install --all-extras         # all features
```

Set the `PYTHONPATH` to resolve local imports when running from source:

```shell
export PYTHONPATH="$PYTHONPATH:src"
```

## Verify the Installation

Check the CLI is available:

```shell
python -m fstg_toolkit --version
```

## Optional Dependencies

| Extra | What it enables | Additional requirement |
|-------|----------------|------------------------|
| `dashboard` | Interactive Dash web dashboard | — |
| `plot` | `multipartite_plot`, `spatial_plot`, `temporal_plot` | — |
| `frequent` | Frequent subgraph pattern mining via SPMiner | [Docker](https://docs.docker.com/get-docker/) must be installed and running |
| `docs` | Build this documentation | — |
