# Spatio-Temporal Graphs

## What is a Spatio-Temporal Graph?

A **spatio-temporal graph (STG)** is a directed graph that models how brain connectivity evolves over time. 
Each node in the graph represents a **set of brain areas at a specific time point**, and edges encode two 
distinct types of relationships:

- **Spatial edges** — connect two sets of brain areas *at the same time point* and *across regions* when their 
  correlation exceeds the threshold. They represent synchronous functional connectivity.
- **Temporal edges** — connect a set of areas across consecutive time points, labeled with an RC5 transition type. 
  They encode how connectivity patterns *change* over time.

This representation unifies the spatial structure of brain connectivity with its temporal dynamics
into a single graph object, enabling both snapshot-level and longitudinal analyses. The brain areas can be grouped
in sets, represented as nodes, when they belong to the same user-defined region.

## Graph Structure

Formally, a spatio-temporal graph `G = (V, E_s ∪ E_t)` where:

- `V` is the set of nodes. Each node `v = (area, time)` uniquely identifies a set of brain areas at a
  given time step.
- `E_s ⊆ V × V` is the set of **spatial edges**. An edge `(u, v) ∈ E_s` exists when both nodes
  share the same time step and their correlation exceeds the configured threshold.
- `E_t ⊆ V × V` is the set of **temporal edges**. An edge `(u, v) ∈ E_t` connects nodes at
  consecutive time steps and carries an [`RC5`](rc5_algebra.md) transition label.

## The `SpatioTemporalGraph` Class

In fSTG Toolkit, STGs are represented by the
{py:class}`fstg_toolkit.graph.SpatioTemporalGraph` class, which extends
`networkx.DiGraph`. This means the full NetworkX API is available for querying and
manipulating the graph.

Key graph-level attributes stored in `graph.graph`:
- `min_time` — the index of the first time step
- `max_time` — the index of the last time step
- `areas` — the areas/regions parcellation as a {py:class}`pandas.DataFrame`

Node attributes include:
- `t` — the time step index
- `areas` — the set of brain area identifiers
- `region` — the region this area belongs to
- `internal_strength` — the calculated internal strength of the set of brain areas subgraph
- `efficiency` — the calculated efficiency of the set of brain areas subgraph

Edge attributes include:
- `type` — `"spatial"` or `"temporal"`
- `transition` — the {py:class}`fstg_toolkit.graph.RC5` transition label (temporal edges only)
- `correlation` — the Pearson correlation value (spatial edges only)
- `t` — the time step index of the origin of the edge

## From Correlation Matrices to Graphs

The factory function
{py:func}`fstg_toolkit.factory.spatio_temporal_graph_from_corr_matrices` converts
a sequence of correlation matrices into an STG:

1. For each time step `t`, the `(N × N)` correlation matrix is thresholded. Connected component subgraphs
   are grouped into sets of areas to form the nodes. Pairs of such nodes whose absolute correlation exceeds the 
   threshold are connected by a spatial edge.
2. For each pair of consecutive time steps `t` and `t+1`, the connectivity patterns of each
   set of areas are compared and an RC5 transition label is assigned to the temporal edge.

Building is parallelized across subjects via `ProcessPoolExecutor`.

## Use Cases

Spatio-temporal graphs are well suited for:

- **Longitudinal analysis** — tracking how connectivity reorganises across sessions,
  disease stages, or experimental conditions.
- **Group comparison** — comparing graph metrics between patient groups and healthy controls.
- **Pattern discovery** — finding recurring subgraph patterns that appear across multiple subjects
  using frequent subgraph mining.
