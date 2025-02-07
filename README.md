[![pipeline status](https://git.unistra.fr/jpontabry/mos-t_fmri/badges/main/pipeline.svg)](https://git.unistra.fr/jpontabry/mos-t_fmri/-/commits/main) 

# MoS-T_fMRI

This package allows you to build, plot, and simulate spatio-temporal graphs for fMRI data. This readme describes briefly the usage of the CLI tool. In the following sections, the installation, the usage and an example are detailed.

## Installation

To install the required dependencies, run:

```sh
conda env create -n <env_name> -f environment.yml
conda activate <env_name>
pip install -r requirements.txt
```

## Usage

The CLI tool provides several commands grouped under `build`, `plot`, and `simulate`. The help for each command is available through the command line using the `--help` option. Some examples and explanations are given in the next section.

## Examples

### Build a Graph

Let be the timeseries of correlations matrices be stored a numpy pickle file `matrices.npz` or `matrices.npy` and the definitions of the areas and regions csv file `areas.csv`.

The areas/regions definition must give the information as in the following table.

| Id_Area | Name_Area | Name_Region |
|---------|-----------|-------------|
| 1       | Area1     | Region1     |
| 2       | Area2     | Region1     |
| 3       | Area3     | Region2     |
| 4       | Area4     | Region3     |

The csv file is then formatted accordingly as follows.

```csv
Id_Area,Name_Area,Name_Region
1,Area1,Region1
2,Area2,Region1
3,Area3,Region2
4,Area4,Region3
```

To build a spatio-temporal graph from the inputs and save the graph to the archive file `my_graph.zip`, use the command:

```sh
python -m fmri_st_graph build matrices.npz areas.csv -o my_graph.zip
```

### Plot a Graph

To plot a graph stored in the file `my_graph.zip` to a dynamic plot, use the command:

```sh
python -m fmri_st_graph plot my_graph.zip dynamic
```

Other available plot types can be accessed through the command `spatial`, `command` and `multipartite` (avoid this last one for large graphs, because of memory issues).

### Simulate a Pattern

To simulate a pattern, give the description of networks across time, spatial and temporal edges. The string syntax for a single network is `area_range,region,internal_strength`, where the area range is defined either by a single area id or by a range between two ids separated by a colon. The description of multiple networks at a given time are concatenated with spaces as separation. A `/` symbol separates networks of two different time instants. The whole description must be surrounded by quotes.

The syntax for a single spatial edge is `network1_id,network2_id,correlation`. Multiple descriptions are concatenated between quotes and separated by spaces.

The syntax for a single temporal edge is `network_id_range,network_id_range`, where the range can be either a single network id, or multiple ids. In the latter, separate the ids with a `-` character. The kind of edges is automatically inferred. For instance, `id,id` means an equal edge while `id-id,id` means a merge and `id,id-id` means a split. Multiple descriptions are concatenated between quotes and separated by spaces.

As an example, the command
```sh
python -m fmri_st_graph simulate -o pattern.zip pattern "1:3,1,0.8 4:5,2,-0.8 / 1:2,1,0.7 3,1,1 4:5,2,-0.8" "1,2,0.6 3,5,0.5" "1,3-4 2,5"
```
create the pattern depicted in the following multipartite plot.

![Example a generated pattern](doc/simulation_pattern_example.pdf "Example a generated pattern")

### Simulate a Sequence

A graph can be simulated from a sequence of pre-generated patterns. The description of the sequence is made of space-separated element, which can be either a pattern, designated as `p<n>` where `n` is the order of the pattern passed to the command, or a number `d`, to create `d` steady states.  

For instance, to create a graph from three distinct patterns and with some steady state in-between patterns, use the command: 

```sh
python -m fmri_st_graph simulate -o sequence.zip sequence pattern1.zip pattern2.zip pattern3.zip "p2 10 p3 5 p1"
```

### Simulate Correlations

To simulate a timeseries of correlation matrices from a spatio-temporal graph, stored in the file `my_graph.zip`, and with a correlation threshold of 0.5, use the command:

```sh
python -m fmri_st_graph simulate -o correlations.npz correlations my_graph.zip -t 0.5
```

The output timeseries matrices will be saved in numpy-compatible format. To set the output path, use the option `-o` of the simulation command. 

## License

TODO
