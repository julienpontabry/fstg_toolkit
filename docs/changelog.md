# Changelog

All notable changes to fSTG Toolkit are documented here.

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
