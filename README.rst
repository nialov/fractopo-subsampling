The main fracture network subsampling functionality has been integrated
into ``fractopo``. The functionality there should be the most up-to-date
and easier to use.

-  https://github.com/nialov/fractopo

This repo has been archived as of 13.3.2026.

Documentation
=============

Though this is somewhat structured like a Python package this more so
a collection of scripts and utilities to conduct fracture network
subsampling and to recreate the environment.

See `SUBSAMPLING_README.rst <SUBSAMPLING_README.rst>`__ for guide on how
to install and conduct subsampling as described in the manuscript.

Running tests
~~~~~~~~~~~~~

To run pytest in currently installed environment:

.. code:: bash

   pipenv run pytest

To run full extensive test suite:

.. code:: bash

   pipenv run invoke test

To run continuous integrations tests locally suite:

.. code:: bash

   pipenv run invoke ci-test


Building docs
~~~~~~~~~~~~~

Docs can be built locally to test that ReadTheDocs can also build them:

.. code:: bash

   pipenv run invoke docs

Invoke usage
~~~~~~~~~~~~

To list all available commands from `tasks.py`:

.. code:: bash

   pipenv run invoke --list

Development
-----------

Development dependencies include:

   -  invoke
   -  nox
   -  copier
   -  pytest
   -  coverage
   -  sphinx

Big thanks to all maintainers of the above packages!


License
-------

Copyright © 2021, Nikolas Ovaskainen.
