[![pipeline status](https://git.unistra.fr/jpontabry/mos-t_fmri/badges/main/pipeline.svg)](https://git.unistra.fr/jpontabry/mos-t_fmri/-/commits/main)

# fSTG Toolkit: an Open-Source Software for Spatio-Temporal Graph Analysis of fMRI data

## Overview

**fSTG Toolkit** is an open-source software dedicated to longitudinal analysis of brain connectivity, modeling data as spatio-temporal graphs. It enables the study of dynamics and reorganization of brain regions, primarily using functional MRI (fMRI) data, but is also compatible with any type of connectivity data.

Main features:
- Building of spatio-temporal graphs from correlation matrices and region definitions.
- Advanced graph metrics computation.
- Interactive visualization of results.
- Simulation of connectivity patterns and sequences.

## Installation

The easiest way to ge started is to create a new environment with the required python and poetry binaries. Using conda, to install the environment and activate it, run:
```shell
conda env create -n <env_name> -f environment.yml
conda activate <env_name>
```

Then in the project's root folder to install the dependencies, run:
```shell
poetry install
```

## Usage

The CLI tool provides several commands building, calculating metrics, plotting, simulating and viewing the results. To see the complete list of commands, run:
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
python -m fstg_toolkit build -o my_graph.zip areas.csv matrices.npz
```

The `build` command also works with multiple sequences of matrices. All sequences stored in a single `.npz` or `.npy` will be red. To build sequences from multiple file, just input them all:
```shell
python -m fstg_toolkit build -o my_graphs.zip areas.csv matrices-1.npz matrices-2.npz matrcices-3.npz
```

### Calculate metrics

Metrics can be calculating using the `metrics` command. From a dataset of built spatio-temporal graph, run:
```shell
python -m fstg_toolkit metrics my_graphs.zip
```

The calculated metrics will be inserted in the dataset archive.

### View the results

To visualize a dashboard to explore the processed data from a dataset with the `show` command, run:
```shell
python -m fstg_toolkit show my_graphs.zip
```

It will start a local server and open a web browser containing the dashboard, that includes the content of the dataset, the raw matrices, a visualization of the spatio-temporal graphs, etc.

[//]: # (### Plot a Graph)

[//]: # ()
[//]: # (To plot a graph stored in the file `my_graph.zip` to a dynamic plot, use the command:)

[//]: # ()
[//]: # (```sh)

[//]: # (python -m fstg_toolkit plot my_graph.zip dynamic)

[//]: # (```)

[//]: # ()
[//]: # (Other available plot types include `spatial`, `command`, and `multipartite` &#40;avoid this last one for large graphs due to memory issues&#41;.)

[//]: # ()
[//]: # (Below is an examples of spatial plot.)

[//]: # (![Example of spatial plot]&#40;doc/plot_spatial_example.png "Example of spatial plot"&#41;)

[//]: # ()
[//]: # (Below is an example of temporal plot.)

[//]: # (![Example of temporal plot]&#40;doc/plot_temporal_example.png "Example of temporal plot"&#41;)

[//]: # ()
[//]: # (### Simulate a Pattern)

[//]: # ()
[//]: # (To simulate a pattern, provide the description of networks across time, spatial, and temporal edges. The string syntax for a single network is `area_range,region,internal_strength`, where the area range is defined either by a single area ID or by a range between two IDs separated by a colon. Descriptions of multiple networks at a given time are concatenated with spaces. A `/` symbol separates networks of two different time instants. The whole description must be surrounded by quotes.)

[//]: # ()
[//]: # (The syntax for a single spatial edge is `network1_id,network2_id,correlation`. Multiple descriptions are concatenated between quotes and separated by spaces.)

[//]: # ()
[//]: # (The syntax for a single temporal edge is `network_id_range,network_id_range`, where the range can be either a single network ID or multiple IDs separated by a `-` character. The kind of edges is automatically inferred. For instance, `id,id` means an equal edge, `id-id,id` means a merge, and `id,id-id` means a split. Multiple descriptions are concatenated between quotes and separated by spaces.)

[//]: # ()
[//]: # (Example command:)

[//]: # ()
[//]: # (```sh)

[//]: # (python -m fstg_toolkit simulate -o pattern.zip pattern "1:3,1,0.8 4:5,2,-0.8 / 1:2,1,0.7 3,1,1 4:5,2,-0.8" "1,2,0.6 3,5,0.5" "1,3-4 2,5")

[//]: # (```)

[//]: # ()
[//]: # (This creates the pattern depicted in the following multipartite plot:)

[//]: # ()
[//]: # (![Example of a generated pattern]&#40;doc/simulation_pattern_example.png "Example of a generated pattern"&#41;)

[//]: # ()
[//]: # (### Simulate a Sequence)

[//]: # ()
[//]: # (A graph can be simulated from a sequence of pre-generated patterns. The sequence description consists of space-separated elements, which can be either a pattern &#40;`p<n>`, where `n` is the order of the pattern passed to the command&#41; or a number &#40;`d`&#41; to create `d` steady states.)

[//]: # ()
[//]: # (Example command:)

[//]: # ()
[//]: # (```sh)

[//]: # (python -m fstg_toolkit simulate -o sequence.zip sequence pattern1.zip pattern2.zip pattern3.zip "p2 10 p3 5 p1")

[//]: # (```)

[//]: # ()
[//]: # (### Simulate Correlations)

[//]: # ()
[//]: # (To simulate a timeseries of correlation matrices from a spatio-temporal graph stored in `my_graph.zip` with a correlation threshold of 0.5, use the command:)

[//]: # ()
[//]: # (```sh)

[//]: # (python -m fstg_toolkit simulate -o correlations.npz correlations my_graph.zip -t 0.5)

[//]: # (```)

[//]: # ()
[//]: # (The output timeseries matrices will be saved in a numpy-compatible format. Use the `-o` option to set the output path.)

[//]: # (## License)
[//]: # ()
[//]: # (TODO)