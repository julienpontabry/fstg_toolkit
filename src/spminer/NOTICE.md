# Provenance and licensing of SPMiner

This directory builds the Docker service that fSTG-Toolkit uses to mine
frequent subgraph patterns. It is driven from Python by
`fstg_toolkit.frequent.SPMinerService`, which builds the image from this
directory and runs `batch_process.py` in a container.

It bundles files of three different origins, listed below.

## Written at ICube, under the CeCILL-2.1 license

`batch_process.py`, `Dockerfile`, `docker-compose.yml`, `requirements.txt`,
`.env`

Copyright 2025 ICube (University of Strasbourg - CNRS), authored by Assaad
ZEGHINA with contributions by Julien PONTABRY. Distributed under the CeCILL
Free Software Agreement v2.1, like the rest of fSTG-Toolkit (see `LICENSE` at
the root of the repository).

## Derived from SPMiner, extended at ICube

`subgraph_mining/decoder.py`, `common/data.py`

Based on the SPMiner sources described below, extended at ICube to support
multi-graphs (spatial and temporal edge types, per-graph input and output
paths, loading of a preprocessed multi-graph dataset). ICube claims
authorship of those modifications only. Because the underlying work carries
no license, these files cannot be redistributed under CeCILL or under any
other license granted by ICube.

## Bundled from SPMiner, unchanged

The remaining files under `common/`, `subgraph_matching/` and
`subgraph_mining/`, plus the pretrained checkpoint `ckpt/model.pt`.

Taken from <https://github.com/snap-stanford/neural-subgraph-learning-GNN>
(Ying et al.). That repository carries no license file, so its authors retain
all rights over these files.
