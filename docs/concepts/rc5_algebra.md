# RC5 Interval Algebra

## Overview

fSTG Toolkit uses the **RC5 (Region Connection Calculus, 5 relations)** to encode how the
connectivity of a brain area changes between two consecutive time steps. Each temporal edge in a
spatio-temporal graph carries an RC5 label that concisely describes the topological relationship
between two connectivity patterns.

RC5 is a subset of Allen's interval algebra, restricted to five mutually exclusive and jointly
exhaustive relations between two regions (connectivity patterns).

## The Five Relations

Let **A** and **B** be the sets of brain areas connected to a given region at time steps `t` and
`t+1` respectively.

| Relation | Symbol | Meaning | Connectivity change |
|----------|--------|---------|---------------------|
| Equal | `EQ` | A = B | The connectivity pattern is unchanged |
| Proper Part | `PP` | A ⊂ B | Connectivity gained new areas (growth) |
| Proper Part (inverse) | `PPi` | B ⊂ A | Connectivity lost areas (shrinkage) |
| Partial Overlap | `PO` | A ∩ B ≠ ∅, A ≠ B | Some areas retained, some gained, some lost |
| Disconnected | `DC` | A ∩ B = ∅ | Complete reorganisation; no overlap |

## Interpretation for Brain Connectivity

These relations provide a semantically meaningful vocabulary for describing connectivity
dynamics:

- **EQ** — The brain area maintains exactly the same functional connections across the two time
  points. This indicates a stable connectivity state.
- **PP** — The area's connectivity pattern grows: all previously connected areas remain, and
  additional areas become connected. This may signal network expansion or recruitment.
- **PPi** — The connectivity pattern shrinks: some connections are lost while none are gained.
  This may reflect network pruning or disengagement.
- **PO** — The pattern partially reorganises: some connections are retained while others are
  replaced. This is the most common relation in dynamic data.
- **DC** — The connectivity pattern undergoes complete reorganisation with no overlap to the
  previous time step. This is rare and may indicate a state transition or noise.

## Usage in Code

RC5 transitions are represented by the {py:class}`fstg_toolkit.graph.RC5` enum:

```python
from fstg_toolkit.graph import RC5

# Accessing a relation
print(RC5.EQ)    # RC5.EQ
print(RC5.PPi)   # RC5.PPi

# Lookup by name
rc5 = RC5.from_name("PO")  # → RC5.PO

# Iterate over all relations
for relation in RC5:
    print(relation)
```

Temporal edges in a {py:class}`fstg_toolkit.graph.SpatioTemporalGraph` carry the RC5 label
in their `rc5` attribute:

```python
for u, v, data in graph.edges(data=True):
    if data.get('type') == 'temporal':
        print(f"{u} → {v}: {data['rc5']}")
```

## References

- Randell, D. A., Cui, Z., & Cohn, A. G. (1992). A spatial logic based on regions and connection.
  *Proceedings of the 3rd International Conference on Knowledge Representation and Reasoning*,
  165–176.
