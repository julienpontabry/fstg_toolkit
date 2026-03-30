# Data Model

## ZIP Archive Format

fSTG Toolkit stores datasets as **ZIP archives** (`.zip` files). Each archive is self-contained
and can hold one or more spatio-temporal graphs together with their associated data.

### Archive Contents

```
my_graphs.zip
├── areas.csv                    # Brain area/region definitions
├── matrices/
│   ├── subject1.npz             # Raw correlation matrices (optional)
│   └── subject2.npz
├── graphs/
│   ├── subject1.json            # Serialised spatio-temporal graph
│   └── subject2.json
├── metrics/
│   ├── local.csv                # Spatial metrics (per node, per time step)
│   └── global.csv               # Temporal metrics (per graph)
└── patterns/
    ├── pattern_0.json           # Frequent subgraph patterns (if mined)
    └── pattern_1.json
```

Files are only written when the corresponding processing step has been run. A freshly built
archive will contain only `areas.csv`, `matrices/`, and `graphs/`.

## Graph JSON Format

Each graph is serialised as a JSON file using the NetworkX `node_link_data` format with
additional fSTG-specific metadata:

```text
{
  "graph": {"max_time": 9, "areas": [1, 2, 3]},
  "nodes": [
    {"id": "(1, 0)", "time": 0, "area": 1, "region": "Visual"},
    ...
  ],
  "links": [
    {"source": "(1, 0)", "target": "(2, 0)", "type": "spatial", "correlation": 0.72},
    {"source": "(1, 0)", "target": "(1, 1)", "type": "temporal", "rc5": "EQ"},
    ...
  ]
}
```

## DataLoader and DataSaver

Two classes manage reading from and writing to archives:

### `DataLoader`

{py:class}`fstg_toolkit.io.DataLoader` provides lazy and eager loading:

```python
from fstg_toolkit.io import DataLoader

loader = DataLoader('my_graphs.zip')

# List available graphs without loading them
names = loader.lazy_load_graphs()

# Load the areas description
areas = loader.load_areas()

# Load a specific graph
graph = loader.load_graph(areas, 'subject1')

# Load metrics
metrics = loader.load_metrics()

# Load frequent patterns
patterns = loader.load_frequent_patterns()
```

### `DataSaver`

{py:class}`fstg_toolkit.io.DataSaver` accumulates data in memory and writes it to a
ZIP archive atomically:

```python
from fstg_toolkit.io import DataSaver

saver = DataSaver()
saver.add_areas(areas_df)
saver.add_graphs({'subject1': graph1, 'subject2': graph2})
saver.add_matrices({'subject1': matrices_array})

total_files, file_descriptions = saver.save('output.zip')
```

Calling `save()` on an **existing** archive merges new data into it rather than
overwriting the whole file.

## Public API Functions

The top-level module exports two convenience wrappers for single-graph workflows:

```python
from fstg_toolkit import save_spatio_temporal_graph, load_spatio_temporal_graph

# Save a single graph (creates a ZIP archive with one graph)
save_spatio_temporal_graph(graph, 'my_graph.zip')

# Load a single graph (raises if the archive contains more than one)
graph = load_spatio_temporal_graph('my_graph.zip')
```

## Data Flow

```
areas.csv + matrices.npz
        │
        ▼
spatio_temporal_graph_from_corr_matrices()   ← factory.py
        │  SpatioTemporalGraph objects
        ▼
DataSaver.save('output.zip')                 ← io.py
        │  ZIP archive
        ▼
DataLoader('output.zip')                     ← io.py
        │
        ├─ calculate_spatial_metrics()       ← metrics.py
        ├─ calculate_temporal_metrics()      ← metrics.py
        └─ SPMinerService.run()              ← frequent/spminer.py
```
