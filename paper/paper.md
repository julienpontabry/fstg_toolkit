---
title: 'fSTG-Toolkit: a toolkit for modeling, processing, and analyzing the longitudinal reorganization of brain connectivity data using spatio-temporal graph methods'
tags:
  - Python
  - graph
  - spatio-temporal
  - fMRI
  - neuroimaging
  - brain connectivity
  - dynamic functional connectivity
  - visualization
authors:
  - given-names: Julien
    surname: Pontabry
    orcid: 0000-0001-7412-4645
    corresponding: true
    affiliation: 1
  - given-names: Assaad
    surname: Zeghina
    orcid: 0000-0002-2649-5609
    affiliation: 1
  - given-names: Antoine
    surname: Vacavant
    orcid: 0000-0001-9616-3282
    affiliation: 2
  - given-names: Aurélie
    surname: Leborgne
    orcid: 0000-0001-8456-2745
    affiliation: 1
affiliations:
  - name: ICube Laboratory -- University of Strasbourg
    index: 1
    ror: 00k4e5n71
  - name: Institut Pascal -- University of Clermont Auvergne
    index: 2
    ror: 03vgfxd91
date: 1 September 2026
bibliography: paper.bib
---


# Summary

A functional MRI session is usually reduced to a single connectivity matrix per subject, which averages away the very thing longitudinal studies want to observe: how connectivity reorganizes. Some methods use sliding window techniques to produce a sequence of matrices for many timepoints, but the actual reorganization between timepoints is often overlooked. fSTG-Toolkit turns a sequence of connectivity matrices into a single graph object that keeps the time axis. This spatio-temporal graph is composed of three families of elements:

- nodes are groups of mutually correlated brain areas inside an anatomical region, at a given timepoint;
- spatial edges link two such groups at a given timepoint;
- temporal edges link the same group between timepoints and carry a label describing the temporal change.

In particular, the labels describe how the groups reorganized: unchanged (EQ), grown (PP), shrunk (PPi), partially reorganized (PO) and groups without overlap between timepoints are disconnected (DC) and simply not linked by a temporal edge.

fSTG-Toolkit provides everything necessary to build such graph and analyze it: spatial and temporal graph metrics, frequent pattern mining across subjects and interactive visual exploration. It is written in pure Python, built upon NetworkX [@hagberg2008networkx] and installable from PyPI. It is usable through a command-line tool, a Python API, or a web dashboard.


# Statement of need

Dynamic connectivity studies aim to describe the reorganization of the cerebral connectivity, usually through a time frame. However, the two main approaches do not provide clear insights:

- static pipelines integrate the time axis into one correlation matrix;
- dynamic pipelines produces a sequence of brain states using sliding windows and clustering [@hutchison2013dynamic; @allen2014tracking; @preti2017dynamic].

In neither case the local reorganization between timepoints is clearly established, and it has to be reconstructed by hand from state labels, if it can be recovered at all.

In practice, pipelines in labs are usually composed of a manual glueing of python libraries: NetworkX, Nilearn, pandas, etc. The workflows are usually built for one particular usage or study and they are rarely shared. It impacts their reproductibility across cohorts.

The toolkit presented in this article implements a serializable longitudinal connectivity graph whose temporal edges carry an interpretable and mutually exclusive relation vocabulary. Because the reorganization is encoded in the structure of the graph, it becomes directly measurable and minable: 

- reorganization rate and the distribution of transition types;
- burstiness and memory reorganization events, borrowed from temporal network analysis [@goh2008burstiness; @holme2012temporal];
- subgraph patterns recurring across subjects, using Multi-SPMiner [@zeghina2023multispminer].

The purpose of fSTG-Toolkit is to widen the available tools to study reorganization in network neurosciences [@bassett2017network]. While the primary focus of the toolkit is the analysis of brain connectivity data, it only requires as input a sequence of correlation matrices and a parcellisation table. Therefore, any connectivity data complying with this contract could be analyzed and visualized.

Note that this toolkit does not include a preprocessing pipeline: the raw BOLD signal must be processed in the standard ways before use. Also, the main focus being the reorganization dynamic, it does not implement any standard graph theory method.


# State of the field

The Brain Connectivity Toolbox [@rubinov2010bct] and the GUI-driven toolboxes built around it (e.g. GraphVar [@kruschwitz2015graphvar], GRETINA [@wang2015gretna] or BRAPH [@mijalkov2017braph]) are mature and have exhaustively documented metric catalogues that set the field standard for static analyzes. However, they rely on a single connectivity matrix as unit of analysis. Time is accounted for only as a repeated and independent analyzes, and the results must be compared externally. For instance, BRAPH does support longitudinal comparison of the same subjects across time points, but it does not model the transitions as a structure of the graph.

For capturing temporal variability, sliding-window methods like ICA with k-means clustering pipelines [@allen2014tracking] and toolboxes such as DynamicDB [@liao2014dynamicbc] are field standard. Their limit originates from the association of one label per window: the local topological relations between successive windows are discarded and the results are highly dependent on the window length and the number of clusters (see for instance [@preti2017dynamic]).

fSTG-Toolkit does not replace the cited software, it complements them. To the authors' knowledge, it is the first openly available toolkit for brain connectivity that

1. encodes inter-timepoints reorganization as typed graph structure using a region connection algebra and
2. ships an end-to-end software from matrices to graph, to metrics, to frequent patterns mining and to interactive exploration in one tool.

Furthermore, the software libraries cited above are mostly MATLAB code, while fSTG-Toolkit has a pip-installable pure Python stack that lowers the barrier and naturally composes with the scientific Python ecosystem.


# Software design

Each correlation matrix of brain areas is thresholded into a graph from a user-defined threshold. Then for each matrix, the connected components are collapsed within a user-defined region of intereset into a single node. This double parcellation is what makes the transitions algebra definable: the relations between sets of brain areas, like EQ, PP, PPi, PO and DC have a meaning in that context. As a counterpart, the individual areas are no longer represented in the graph's structure, despite the information is preserved in each node's attribute. The concept is illustrated in \autoref{fig:concept}(A-B).

![Overview of the spatio-temporal graph model. (A) At each time point the correlation matrix is thresholded. (B) Connected components of areas within each user-defined region form the networks that become the nodes of the graph. (C) Consecutive time points are linked by temporal edges labelled with the RC5 relation between the two sets of areas; a network with no overlap at the next time point (DC) simply has no outgoing temporal edge, as for {A8} towards {A6}.\label{fig:concept}](figures/concept.png)

The RC5 model has been chosen as temporal transition algebra because it offers five exhaustive and mutually exclusive relations [@randell1992]. The five modeled transitions are: EQ for no change, PP for grown, PPi for shrunk, PO for partial reorganization and DC for the absence of relation. In the graph model, DC is represented as the absence of edge rather than a labelled one. With this transition vocabulary, a neuroscientist can read the meaning directly from its visualization and a subgraph mining algorithm can operate on its finite states.  However, the quantization is lost: a change of one area or twenty areas will be both labeled with the same transition. An example is illustrated in \autoref{fig:concept}(C).

A representation of a spatio-temporal graph is implemented as a subclass of the `DiGraph` class of the NetworkX package. This choice makes available the entire package's ecosystem available and encompasses both representation of spatial and temporal edges, at a cost of storing twice the spatial edges. For instance, it allows to implements the metrics usually in a few lines without having to convert between structures.

The core of the toolkit depends only on NetworkX, NumPy, pandas and Click packages. Opt-in extras are available to give more functionalities like plotting, dashboard and frequent pattern mining. The later rely on the Multi-SPMiner software [@zeghina2023multispminer] that is shipped alongside with fSTG-Toolkit as a component. When requested, it is run inside a Docker container to avoid dependencies conflicts with the toolkit. As a counterpart, frequent patterns mining requires a running Docker daemon and the first run is slowed by the built of the Docker image.

An important component of the toolkit, accessible with the dashboard extra dependancy group, is the visualization dashboard. It allows to explore interactively the spatio-temporal graphs, the metrics and the mined frequent patterns. Two flavours are provided: (i) a local dashboard that opens a result archive to explore it, or (ii) a full remote service on the local network for internal usage of small teams, usable from a web browser and without any installation, that allows a user to upload a dataset, start its processing and come back later to interact with the dashboard of the result. A capture of the spatio-temporal graph view in the dashboard is shown in \autoref{fig:dashboard}.

![The interactive dashboard, showing a subject's spatio-temporal graph alongside the metrics derived from it.\label{fig:dashboard}](figures/dashboard.png)

As the metrics are implemented as an extandable decorator-based metric registry, the users can therefore register their own metrics on the fly, without needing to fork the entire repository.

To produce a traceable and a sharable results file, all artifacts (raw matrices, graphs in JSON format, metrics and mined patterns) are stored in a single ZIP archive. New artifacts can be added without changing the container format, using the data registry and the data handler protocol.

Finally, the toolkit is extensively tested, with more than four thousand of lines of tests plus executed doctests, run in continuous integration on Linux and macOS across Python 3.12 and 3.13. However, the Dash callbacks are only tested manually rather than by automated tests.


# Research impact statement

The frequent pattern mining component is the reference implementation of the authors' published spatio-temporal graph mining research: Multi-SPMiner [@zeghina2023multispminer] and continuated in DeepQMiner [@zeghina2025deepqminer]. See [@zeghina2024review] for a landscape survey of this research area. By encapsulating in a software the outcome of this research, fSTG-Toolkit is what makes that work usable on brain connectivity data by researches who are not graph-mining specialists.

The toolkit has been presented to the community during a live software demonstration at the 2026 IEEE ISBI conference [@pontabry2026isbidemo]. The source code is available on [GitHub](https://github.com/julienpontabry/fstg_toolkit) and has been released and published on [PyPI](https://pypi.org/project/fSTG-Toolkit/). The full documentation for concepts, installation and usage is accessible on its [ReadTheDoc](https://fstg-toolkit.readthedocs.io/en/latest/) page.

The toolkit is being adopted in the author lab to complete the set of tools used to analyze the dynamic of brain connectivity data. It is planned to support, at least partly, future studies on the dynamic reorganization of brain functionnal connectivity.

While it has been built with neurosciences in mind, the only assumptions on the input are sequences of correlation matrices and a parcellation table. Therefore, other applications outside of neurosciences are under study: for instance it could be used to study the reorganization of territorial data. 


# AI usage disclosure

Generative AI assistance was used in preparing the software documentation and in drafting this manuscript. Its use in the source code was limited to occasional targeted assistance in resolving specific implementation problems. The problem framing, the spatio-temporal graph and RC5 data model, and all architectural decisions presented in this article are the authors' own. The authors have reviewed and tested all code and documentation and take full responsibility for the content and claims of this work.


# Acknowledgements

The authors would like to thank the ANR MoS-T project (grant ANR-21-CE23-0015), which funded the research this toolkit builds on. Céline MEILLIER is thanked for her insight on the MoS-T project and the toolkit. The IMIS research team at ICube Laboratory is also thanked for its testing of the software from the user perspective. Laetitia DEGIORGIS is specially thanked for providing a dataset of functionnal connectivity of the mouse model that has been used during the development of the toolkit.


# References


