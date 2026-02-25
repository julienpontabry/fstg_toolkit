# Copyright 2025 ICube (University of Strasbourg - CNRS)
# author: Julien PONTABRY (ICube)
#
# This software is a computer program whose purpose is to provide a toolkit
# to model, process and analyze the longitudinal reorganization of brain
# connectivity data, as functional MRI for instance.
#
# This software is governed by the CeCILL-B license under French law and
# abiding by the rules of distribution of free software. You can use,
# modify and/or redistribute the software under the terms of the CeCILL-B
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info".
#
# As a counterpart to the access to the source code and rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty and the software's author, the holder of the
# economic rights, and the successive licensors have only limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading, using, modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean that it is complicated to manipulate, and that also
# therefore means that it is reserved for developers and experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and, more generally, to use and operate it in the
# same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL-B license and that you accept its terms.

import pandas as pd
import plotly.express as px
from plotly.graph_objects import Figure


def areas_per_region_figure(records: list[dict[str, str]]) -> Figure:
    groups = pd.DataFrame.from_records(records, index='Id_Area').groupby('Name_Region')
    df = groups.count()
    df['Areas'] = groups.apply(lambda g: g['Name_Area'].to_list(), include_groups=False)

    return px.bar(df, orientation='h', x='Name_Area', height=600,
                  labels={'Name_Area': 'Number of areas', 'Name_Region': 'Region'},
                  title="Distribution of areas per region", hover_data='Areas')


def subjects_per_factors_figure(records: list[dict[str, str]], factors: list[str]) -> Figure:
    df = pd.DataFrame.from_records(records)[factors + ['Subject']]

    labels = {'Subject': "Number of subjects"}

    if len(df.columns) > 1:
        df = df.groupby(factors).count().reset_index()
    else:
        df = pd.DataFrame(df.count(), columns=["All"]).T
        labels['index'] = ""

    params = dict(zip(('x', 'color'), factors))
    return px.bar(df, y='Subject', **params, height=500,
                  labels=labels,
                  title="Distribution of subjects per factors")
