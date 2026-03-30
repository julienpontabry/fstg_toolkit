![PyPI - Python Version](https://img.shields.io/pypi/pyversions/fSTG-Toolkit)
[![CI](https://github.com/julienpontabry/fstg_toolkit/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/julienpontabry/fstg_toolkit/actions/workflows/ci.yml)
![PyPI - Version](https://img.shields.io/pypi/v/fSTG-Toolkit)
![PyPI - License](https://img.shields.io/pypi/l/fSTG-Toolkit)
![Read the Docs](https://img.shields.io/readthedocs/fSTG-Toolkit)

# fSTG Toolkit: an Open-Source Software for Spatio-Temporal Graph Analysis of fMRI data

## Overview

**fSTG Toolkit** is an open-source software dedicated to longitudinal analysis of brain connectivity, modeling data as spatio-temporal graphs. It enables the study of dynamics and reorganization of brain regions, primarily using functional MRI (fMRI) data, but is also compatible with any type of connectivity data.

Current main features:
- Building of spatio-temporal graphs from correlation matrices and region definitions.
- Advanced graph metrics computation.
- Interactive visualization of results via a web dashboard.
- Simulation of connectivity patterns and sequences.
- Frequent subgraph pattern mining via SPMiner integration.

## Installation

### Installation from PyPI

The easiest way to get started is to install the package from PyPI. Make sure you have a Python environment ready with a supported version (see the badge above), then run:
```shell
pip install fSTG-Toolkit
```

To install optional feature sets:
```shell
pip install "fSTG-Toolkit[dashboard]"   # web dashboard
pip install "fSTG-Toolkit[plot]"        # matplotlib plots
pip install "fSTG-Toolkit[frequent]"    # frequent pattern mining (requires Docker)
pip install "fSTG-Toolkit[dashboard,plot,frequent]"  # everything
```

### Installation from Source

To install from source, create a new environment with the required Python and Poetry binaries. Using conda:
```shell
conda env create -n <env_name> -f environment.yml
conda activate <env_name>
```

Then in the project's root folder, install the dependencies:
```shell
poetry install
```

To install optional feature sets:
```shell
poetry install --extras dashboard   # web dashboard
poetry install --extras plot        # matplotlib plots
poetry install --extras frequent    # frequent pattern mining (requires Docker)
poetry install --all-extras         # everything
```

## Usage

The CLI tool provides several command groups: `graph`, `plot` and `dashboard`. To see the complete list of commands, run:
```shell
python -m fstg_toolkit --help
```

Use the `--help` option with any command to get specific help. Some examples and explanations are provided in the next section.

## Examples

### Build one or multiple graphs

Assume the timeseries of correlation matrices are stored in a numpy pickle file (`matrices.npz` or `matrices.npy`) and the definitions of the areas and regions are in a CSV file (`areas.csv`).

The areas/regions definition must be formatted as follows:

| Id_Area | Name_Area | Name_Region |
|---------|-----------|-------------|
| 1       | Area1     | Region1     |
| 2       | Area2     | Region1     |
| 3       | Area3     | Region2     |
| 4       | Area4     | Region3     |

Accordingly, the CSV file should look like this:

```csv
Id_Area,Name_Area,Name_Region
1,Area1,Region1
2,Area2,Region1
3,Area3,Region2
4,Area4,Region3
```

To build a spatio-temporal graph from the inputs and save the graph to the archive file `my_graph.zip`, use the command:

```shell
python -m fstg_toolkit graph build -o my_graph.zip areas.csv matrices.npz
```

The `build` command also works with multiple sequences of matrices. All sequences stored in a single `.npz` or `.npy` will be read. To build sequences from multiple files, input them all:
```shell
python -m fstg_toolkit graph build -o my_graphs.zip areas.csv matrices-1.npz matrices-2.npz matrices-3.npz
```

A correlation threshold can be set with `-t` (default 0.4):
```shell
python -m fstg_toolkit graph build -o my_graph.zip -t 0.5 areas.csv matrices.npz
```

### Calculate metrics

Metrics can be calculated using the `metrics` command. From a dataset of built spatio-temporal graphs, run:
```shell
python -m fstg_toolkit graph metrics my_graphs.zip
```

The calculated metrics will be inserted in the dataset archive.

### Frequent Pattern Mining

Frequent subgraph pattern mining requires Docker and the `[frequent]` extra. To run the analysis on a dataset:

```shell
python -m fstg_toolkit graph frequent my_graphs.zip
```

The detected frequent patterns will be inserted in the dataset archive and can be explored interactively in the dashboard.

### View the results

To visualize a dashboard to explore the processed data from a dataset with the `show` command, run:
```shell
python -m fstg_toolkit dashboard show my_graphs.zip
```

It will start a local server and open a web browser containing the dashboard, that includes the content of the dataset, the raw matrices, a visualization of the spatio-temporal graphs, etc. An illustration of the dashboard is shown below.

![Illustration of the dashboard.](doc/images/illustration_web-viewer.png)

To run a persistent multi-dataset server, use the `serve` command:
```shell
python -m fstg_toolkit dashboard serve <data_path> <upload_path>
```

#### Factors and Subjects Detection

If the names of the matrices are formatted, factors and subjects will be automatically detected and can be used to filter the data and choose the display of the plots. The parts of the names must be separated either by underscores (`_`) or by slashes (`/`) or a combination of both. For instance, the following names will be correctly parsed:
- `control_time1_T21`;
- `control/time2_T22`;
- `group1_time2/T31`;
- `group1_time1_T11`.

The subjects will be matched to the part that has different values between the names, and the factors will be the parts that are common to multiple names. If a part is similar in all names, it will not be considered. In this case, the subjects are `T21`, `T22`, `T31`, and `T11`, and the factors are `control` and `group1` for first factor, `time1` and `time2` for the second factor.
