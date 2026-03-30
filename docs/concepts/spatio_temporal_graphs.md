# Spatio-Temporal Graphs

## What is a Spatio-Temporal Graph?

A **spatio-temporal graph (STG)** is a directed graph that models how brain connectivity evolves
over time. Each node in the graph represents a **brain area at a specific time point**, and edges
encode two distinct types of relationships:

- **Spatial edges** — connect two brain areas *at the same time point* when their correlation
  exceeds the threshold. They represent synchronous functional connectivity.
- **Temporal edges** — connect a brain area (or group of areas) across consecutive time points,
  labelled with an RC5 transition type. They encode how connectivity patterns *change* over time.

This representation unifies the spatial structure of brain connectivity with its temporal dynamics
into a single graph object, enabling both snapshot-level and longitudinal analyses.

## Graph Structure

Formally, a spatio-temporal graph `G = (V, E_s ∪ E_t)` where:

- `V` is the set of nodes. Each node `v = (area, time)` uniquely identifies a brain area at a
  given time step.
- `E_s ⊆ V × V` is the set of **spatial edges**. An edge `(u, v) ∈ E_s` exists when both nodes
  share the same time step and their Pearson correlation exceeds the configured threshold.
- `E_t ⊆ V × V` is the set of **temporal edges**. An edge `(u, v) ∈ E_t` connects nodes at
  consecutive time steps and carries an [`RC5`](rc5_algebra.md) transition label.

## The `SpatioTemporalGraph` Class

In fSTG Toolkit, STGs are represented by the
{py:class}`fstg_toolkit.graph.SpatioTemporalGraph` class, which extends
`networkx.DiGraph`. This means the full NetworkX API is available for querying and
manipulating the graph.

Key graph-level attributes stored in `graph.graph`:
- `max_time` — the index of the last time step
- `areas` — the list of brain area identifiers

Node attributes include:
- `time` — the time step index
- `area` — the brain area identifier
- `region` — the region this area belongs to

Edge attributes include:
- `type` — `"spatial"` or `"temporal"`
- `rc5` — the {py:class}`fstg_toolkit.graph.RC5` transition label (temporal edges only)
- `correlation` — the Pearson correlation value (spatial edges only)

## From Correlation Matrices to Graphs

The factory function
{py:func}`fstg_toolkit.factory.spatio_temporal_graph_from_corr_matrices` converts
a sequence of correlation matrices into an STG:

1. For each time step `t`, the `(N × N)` correlation matrix is thresholded. Pairs of areas
   whose absolute correlation exceeds the threshold are connected by a spatial edge.
2. For each pair of consecutive time steps `t` and `t+1`, the connectivity patterns of each
   area are compared and an RC5 transition label is assigned to the temporal edge.

Building is parallelised across subjects via `ProcessPoolExecutor`.

## Use Cases

Spatio-temporal graphs are well suited for:

- **Longitudinal analysis** — tracking how connectivity reorganises across sessions,
  disease stages, or experimental conditions.
- **Group comparison** — comparing graph metrics between patient groups and healthy controls.
- **Pattern discovery** — finding recurring subgraph patterns that appear across multiple subjects
  using frequent subgraph mining.
- **Simulation** — generating synthetic datasets with controlled connectivity dynamics for
  pipeline validation.
