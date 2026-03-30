pages
=====

Dash multipage layout definitions. Each module registers itself as a Dash page via
``dash.register_page()`` at import time and defines a ``layout`` function that returns the
page's component tree.

.. note::

   These modules cannot be imported standalone — they must be loaded within an initialised Dash
   application. For this reason their API is not auto-extracted here.

home
----

Path: ``/``

The landing page of the dashboard. Provides an overview of the toolkit and links to loaded
datasets.

list
----

Path: ``/list``

Displays all datasets currently tracked by the server.
Each entry links to the corresponding dashboard view.

submit
------

Path: ``/submit``

Form for uploading new ``.zip`` dataset archives to the server (persistent ``serve`` mode only).

dashboard
---------

Path: ``/dashboard/<token>``

The main interactive analysis view for a single dataset. Contains tabs for data overview,
raw correlation matrices, spatio-temporal graph visualisation, metrics charts, and frequent
subgraph patterns.
