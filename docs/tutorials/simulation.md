# Simulating Graphs and Matrices

fSTG Toolkit provides two simulators for generating synthetic data, useful for testing,
benchmarking, and validating analysis pipelines.

## CorrelationMatrixSequenceSimulator

Given an existing spatio-temporal graph, this simulator regenerates the underlying correlation
matrix sequence that is consistent with the graph's connectivity structure.

```python
from fstg_toolkit import load_spatio_temporal_graph, CorrelationMatrixSequenceSimulator
import numpy as np

# Load an existing graph
graph = load_spatio_temporal_graph('my_graphs.zip', 'subject1')

# Create the simulator
simulator = CorrelationMatrixSequenceSimulator(graph, threshold=0.4)

# Simulate
matrices = simulator.simulate()  # shape: (T, N, N)
print(f"Simulated {matrices.shape[0]} correlation matrices of size {matrices.shape[1]}x{matrices.shape[2]}")

# Save for reuse
np.savez_compressed('simulated.npz', simulated=matrices)
```

### Via the CLI

```shell
python -m fstg_toolkit graph simulate correlations my_graph.zip -t 0.4
```

## SpatioTemporalGraphSimulator

Generates a full spatio-temporal graph from a sequence of pattern graphs. Useful for constructing
controlled synthetic datasets with known connectivity transitions.

```python
from fstg_toolkit import SpatioTemporalGraphSimulator, load_spatio_temporal_graph

# Load pattern graphs
pattern1 = load_spatio_temporal_graph('pattern1.zip')
pattern2 = load_spatio_temporal_graph('pattern2.zip')

# Assemble a simulator with named patterns
simulator = SpatioTemporalGraphSimulator(p1=pattern1, p2=pattern2)

# Simulate: 3 steady states, then pattern p1, then 2 steady states, then pattern p2
graph = simulator.simulate(3, 'p1', 2, 'p2')
```

## Generating Pattern Graphs

The `generate_pattern` function creates a single-step spatio-temporal graph pattern from a
programmatic description:

```python
from fstg_toolkit import generate_pattern

# One network: areas 1 to 3, region 0, internal strength 0.8
networks = [[(  (1, 3), 0, 0.8  )]]

pattern = generate_pattern(networks_list=networks)
```

### Via the CLI

The `graph simulate pattern` command accepts the same description in a compact string format:

```shell
# One network (areas 1–3, region 0, strength 0.8)
python -m fstg_toolkit graph simulate pattern "1:3,0,0.8"

# Two time steps with spatial and temporal edges
python -m fstg_toolkit graph simulate pattern "1:3,0,0.8/1:3,1,0.7" "0,1,0.6" "0,1"
```

The CLI command writes the generated graph to a file (default: `output`). Use `-o` to set a
custom output path:

```shell
python -m fstg_toolkit graph simulate -o my_pattern pattern "1:3,0,0.8"
```

### Generating a Sequence via the CLI

```shell
# Simulate a graph from two patterns: p1 p2 are pattern file paths,
# the sequence is: 5 steady states, pattern1, 3 steady states, pattern2
python -m fstg_toolkit graph simulate sequence pattern1 pattern2 "5 p1 3 p2"
```

## Next Steps

- [Building Graphs](building_graphs.md)
- [API Reference: simulation](../api/simulation.rst)
