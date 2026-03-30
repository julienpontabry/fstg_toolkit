# Changelog

All notable changes to fSTG Toolkit are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.1.0] — 2026-03-30

### Added

- Epilog tips in CLI help messages for optional features when the relevant extras are not installed.
- Faster graph interactions in the dashboard.

### Fixed

- No crash when optional dependencies are not installed at startup.
- Improved handling of epilog tips for extras dependencies.
- GitHub Actions publish workflow.

## [1.0.0] — Initial release

### Added

- `graph build` CLI command to build spatio-temporal graphs from correlation matrices.
- `graph metrics` CLI command to compute spatial and temporal graph metrics.
- `graph simulate` CLI command group for pattern and sequence simulation.
- `graph frequent` CLI command for frequent subgraph pattern mining via SPMiner.
- `dashboard show` and `dashboard serve` CLI commands for the interactive web dashboard.
- `plot` CLI command group for matplotlib-based graph visualisations.
- `SpatioTemporalGraph` class extending `networkx.DiGraph`.
- `RC5` enum encoding temporal transitions (EQ, PP, PPi, PO, DC).
- `DataLoader` and `DataSaver` classes for ZIP-based serialisation.
- `CorrelationMatrixSequenceSimulator` and `SpatioTemporalGraphSimulator` for synthetic data.
- SPMiner Docker integration for frequent subgraph pattern mining.
- CeCILL-B license.
