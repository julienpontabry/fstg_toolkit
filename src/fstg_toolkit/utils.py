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

import logging
import logging.config
from importlib.resources import files
from typing import Optional

import yaml

logger = logging.getLogger()


def setup_logging(level: Optional[str], verbose: bool):
    """Setup logging configuration for the package.

    Parameters
    ----------
    level : str
        The logging level. It should be a lower or upper case string matching a level of the logging package.
    verbose : bool
        The logging verbosity. If true, the logs will appear in the console as well.
    """
    if level is None:
        logger.setLevel(logging.NOTSET)
        return  # no logging in that case

    config_path = files(__package__).joinpath('logging.yml')
    with config_path.open('r') as file:
        config = yaml.safe_load(file)

    config['root']['level'] = level.upper()

    if not verbose:  # console handler used only when verbose
        config['root']['handlers'] = ['file']

    logging.config.dictConfig(config)
