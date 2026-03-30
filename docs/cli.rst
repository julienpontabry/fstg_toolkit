CLI Reference
=============

The fSTG Toolkit CLI is invoked as a Python module:

.. code-block:: shell

   python -m fstg_toolkit [OPTIONS] COMMAND [ARGS]...

The CLI is organised into four command groups: ``graph``, ``plot``, ``dashboard``, and
``simulate`` (a subgroup of ``graph``). The ``plot`` and ``dashboard`` groups require the
``[plot]`` and ``[dashboard]`` extras respectively. The ``graph frequent`` command requires the
``[frequent]`` extra.

Main CLI
--------

.. click:: fstg_toolkit.__main__:cli
   :prog: python -m fstg_toolkit
   :nested: none

graph
-----

.. click:: fstg_toolkit.__main__:graph
   :prog: python -m fstg_toolkit graph
   :nested: full

plot
----

Requires the ``[plot]`` extra (``pip install "fSTG-Toolkit[plot]"``).

.. click:: fstg_toolkit.__main__:plot
   :prog: python -m fstg_toolkit plot
   :nested: full

dashboard
---------

Requires the ``[dashboard]`` extra (``pip install "fSTG-Toolkit[dashboard]"``).

.. click:: fstg_toolkit.__main__:dashboard
   :prog: python -m fstg_toolkit dashboard
   :nested: full
