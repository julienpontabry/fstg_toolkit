# Changelog

All notable changes to fSTG Toolkit are documented here.


## fSTG-Toolkit v1.3.0

### Breaking Changes

- License: The project moved from CeCILL-B to CeCILL-2.1, an OSI-approved license. The
  license headers of all source files and the `LICENSE` file were updated accordingly.

### Features

- Python 3.13 support: The supported Python range is now `>=3.12,<3.14` (the previous
  bound wrongly excluded most 3.13 releases).
- Bundled SPMiner service: The multi-graph SPMiner service moved inside the
  `fstg_toolkit` package so that it ships with the wheels, along with a `NOTICE.md`
  describing the provenance and licensing of its files.
- Community files: Added a citation file (`CITATION.cff`), a code of conduct, a
  contributing guide, issue and pull request templates, and a community section in the
  README and the documentation.

### Bug Fixes

- Graph copy: `SpatioTemporalGraph.copy()` now carries the brain areas over to the
  copied graph, sharing them for a view and duplicating them otherwise.
- SPMiner path: Fixed the path used to locate the bundled SPMiner service.
- CLI simulation output: The `-o/--output_path` option of `graph simulate` is now
  converted to a path object, as the other commands do.
- Package metadata: Fixed a typo in the package description and added the missing
  author of the SPMiner code.

### Documentation

- Clarified the temporal transitions algebra, the data model, and the distinction
  between graph nodes and brain areas.
- Moved the dashboard tutorial after the pattern mining one.
- Fixed the Read the Docs badge and various mistakes in the documentation.
- Added a JOSS draft paper with its figures and bibliography.

### Continuous Integration

- Added unit tests for the visualization helpers and the command line interface.
- Added a coverage job publishing a report summary, an HTML artifact, and a badge.
- Updated the GitLab CI configuration and the repository links after the migration to
  the ICube-Medical-Image-Computing organization.


## fSTG-Toolkit v1.2.1

- Matplotlib dependency possible versions changed to avoid a API break.


## fSTG-Toolkit v1.2.0

### Bug Fixes

- Graph interactions: Fixed a performance issue causing slow graph interactions in the dashboard.
- Optional dependencies: Fixed a crash that occurred when optional dependencies were not installed.
- GitHub Actions: Fixed the publish workflow.

### Features

- Documentation: Added project documentation and logos.
- CLI extras tips: Improved handling of epilog tips for extras dependencies in the CLI.


## fSTG-Toolkit v1.1.0

### Features

- Frequent patterns: New figures for frequent subgraph patterns, including pattern index/count tooltips, co-occurrence heatmaps, and a figure registry
- Configurable graph nodes: Node color and size are now configurable; color scale adapts to data
- Confidence bounds for metrics: Added 95% confidence bounds to plots
- Improved tooltips: Metrics tooltips customized per figure; tooltips adapt to figure context; pattern indices shown in non-pattern tooltips
- Pattern figure descriptions: Added scientific descriptions for pattern figures

### Bug Fixes

- Fixed nodes' size mapping
- Fixed tooltips positioning relative to frequent patterns histograms
- Fixed default tooltip hidden behind custom tooltip on patterns graph
- Fixed integer tick marks on count scales


## fSTG-Toolkit v1.0.0

Initial release with all functionalities:

- spatio-temporal graph building ;
- local and global metrics computation ;
- frequent patterns analysis ;
- dashboard and visualization ;
- dashboard serving.
