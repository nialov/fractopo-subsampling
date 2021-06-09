Future plans and development
============================

The whole process of subsampling can be streamlined but there's many
parts which are opionated and there's no single good way. There's
multiple approaches to using subsamples of fracture data in
understanding uncertainty. More than likely the approach implemented
here is not interesting in all cases when working with fracture data.

Subsampling could be partly implemented in ``fractopo`` to reduce the
complexity of the install and execution. Notebooks could be replaced
with module Python code but the notebooks do contain the most opionated
parts of the subsampling and the implementation of how the subsamples
are analyzed should be the most transparent part of the process
as it is the most important. Notebooks also allow quick experimentation.

How to recreate subsampling environment
=======================================

Note! Only for Linux-based systems with ``bash`` and Python 3.8
installed.

Automatic install
-----------------

Requires that ``kaggle``, ``git``, ``python3``, ``curl`` and
``virtualenv`` cli commands are installed, configured
(https://github.com/Kaggle/kaggle-api#api-credentials) and available on
your system.

1. Create working directory e.g. ``~/projects/subsampling``
2. Run the full install script from ``fractopo-subsampling`` repository
   with the following command:

.. code:: bash

   # As a general advice when running external scripts such as this
   # you should verify the script before executing by visiting the
   # link after the `curl` command below.
   curl https://raw.githubusercontent.com/nialov/fractopo-subsampling/master/subsampling_full_install.sh | bash

3. Done! See steps 9. and later in the *Manual install* section for
   information on how to run subsampling. You do not need to download
   and move the data in manual step 12, all downloading has been done
   automatically.

Manual install
--------------

1. Create working directory e.g. ``~/projects/subsampling``

2. Switch to the working directory.

3. Go to https://www.kaggle.com/nasurz/getaberget-fracture-trace-dataset
   and download the fracture trace dataset.

   -  Alternative to manual downloading is the ``kaggle-api`` Python
      app. If you have it installed (``pip install kaggle``) and
      configured (https://github.com/Kaggle/kaggle-api#api-credentials),
      you can run:

      .. code:: bash

         kaggle datasets download nasurz/getaberget-fracture-trace-dataset

   -  Either way you will have downloaded a zip file with the dataset.

4. Extract files and directory within the zip to the working directory.

5. Create a Python virtual environment.

   -  You may use virtualenv, pipenv or poetry.
   -  Python must be version 3.8.\*

6. Activate the virtual environment.

7. Download some prerequisites for installing
   https://github.com/nialov/fractopo-subsampling

   -  A helper download script is located in the repo. It can be
      executed with the below code.
   -  As general advice for executing such scripts, you should take a
      look at the code yourself before executing to verify its contents
      and to avoid malicious scripts.

   .. code:: bash

      # This below command will download a script from the repo and 
      # execute it directly in bash
      # It will git clone the fractopo-subsampling repo and copy some
      # files (notebooks, python script files, requirements.txt)
      # to the current working directory
      curl https://raw.githubusercontent.com/nialov/fractopo-subsampling/master/subsampling.sh | bash

8. Install ``fractopo-subsampling`` with ``requirements.txt`` to your
   virtual environment of choice.

   .. code:: bash

      # pip install
      pip install -r requirements.txt

      # pipenv install
      pipenv install -r requirements.txt

      # poetry install
      cat requirements.txt | xargs poetry add

9. The environment is ready.

**If you do not wish to conduct new subsampling or base circle
characterization, skip to step 11. to download and use the exact dataset
used in the manuscript.**

Subsampling and base circle characterization have been implemented as
``invoke`` tasks in ``tasks.py``. All invoke tasks can be displayed with
``invoke --list`` command. ``invoke`` should already be installed in the
Python environment.

.. code:: bash

   # if virtualenv is activate
   invoke --list

   # using pipenv
   pipenv run invoke --list

   # using poetry
   poetry run invoke --list

The ``invoke`` tasks will do characterization and subsampling from all
target areas (all rows) that are in ``relations.csv``. To conduct
network analysis of all base circles and store results in jupyter
notebooks for later inspection:

.. code:: bash

   # Prepend invoke with pipenv run or poetry run if using them
   invoke network-all --overwrite --notebooks

To store characterization results in a single GeoPackage as points for
spatial analysis and reference value plotting:

.. code:: bash

   invoke network-all --overwrite --points

To conduct stage 1 subsampling 5 times for each base circle:

.. code:: bash

   invoke network-subsampling --how-many 5

To collect results of stage 1 subsampling (do after stage 1
subsampling):

.. code:: bash

   invoke gather-subsamples

10. Most of the analysis and stage 2 subsampling is in the notebooks in
    the ``notebooks`` directory. The virtual environment should already
    have ``jupyter lab`` installed.

    .. code:: bash

       # Open jupyter lab (execute within the virtual environment!)
       # Should open jupyter lab in your native browser
       jupyter lab

    -  You can download the exact dataset I used with Step 1 and Step 2
       subsampling results as csvs from kaggle as well in step 11 and
       onwards.
    -  Or alternatively repeat the subsampling or base circle
       characterization that is introduced in the previous step to get
       unique subsamples from the same base fracture dataset.
    -  Configure notebook analysis in
       ``notebooks/subsampling_config.py`` and within the notebooks
       themselves.
    -  Notebook ``Base_Circle_Analysis_Figure_7.ipynb`` needs to be run
       before ``Subsampling_Figures_8_9_and_10.ipynb`` to create base
       circle reference value csv.

11. If you wish to use the exact datasets of stage 1 and 2 subsampling
    that I used, continue to 12 to download them. Otherwise, we're done!
    See step 9 for brief introduction to stage 1 subsampling and base
    circle characterization.

12. Go to
    https://www.kaggle.com/nasurz/getaberget-subsampled-fracture-network-dataset
    and download the dataset (two csv files and one GeoPackage).

13. Default path for the stage_1 csv dataset is
    ``results/subsampling/collected/stage_1_subsampling_results.csv``
    and for stage_2 csv dataset
    ``results/subsampling/cached_subsamples/stage_2_aggregated_subsampling_results.csv.``
    Default path for the GeoPackage is
    ``results/Ahvenanmaa_analysis_points.gpkg``. Create the directories
    relative to the current working directory and put the csvs and
    GeoPackage in the default paths.

    .. code:: bash

       # Creating directories and moving the subsampling files
       mkdir results/subsampling/cached_subsamples -p 
       mkdir results/subsampling/collected -p 
       mv stage_1_subsampling_results.csv results/subsampling/collected/
       mv stage_2_aggregated_subsampling_results.csv results/subsampling/cached_subsamples/
       mv Ahvenanmaa_analysis_points.gpkg results/

    -  The csv paths can be alternatively changed within the
       ``notebooks/Subsampling_Figures_8_9_and_10.ipynb`` notebook but
       note that they are relative to the notebook (use ``..`` in paths
       to go to previous directory).

14. You should now be able to exactly replicate the subsampling results
    and plots using the notebooks in ``notebooks`` directory.

    .. code:: bash

       # To open jupyter lab for notebook viewing and execution
       # Prepend with pipenv run or poetry run if using them
       jupyter lab

    -  Notebook ``Base_Circle_Analysis_Figure_7.ipynb`` needs to be run
       before ``Subsampling_Figures_8_9_and_10.ipynb`` to create base
       circle reference value csv.

Subsampling own data
====================

1. Proceed with automatic or manual installation as described in `How to
   recreate subsampling
   environment <#how-to-recreate-subsampling-environment>`__. You can
   skip data downloading steps (12,13) in manual installation. You
   should download the fracture trace data in steps 3 and 4 as you can
   use the ready data as reference when replacing it with your own.

2. Fracture traces must be linked to target areas which define the area
   in which the traces are analyzed. This linking is done with the
   ``relations.csv`` file which comes with the fracture trace data from
   kaggle. Example contents:

   .. code:: csv

      area,traces,thematic,scale,valid,empty,diameter
      Getaberget_20m_1_1_area,Getaberget_20m_1_traces,ahvenanmaa,20m,True,False,50
      Getaberget_20m_1_2_area,Getaberget_20m_1_traces,ahvenanmaa,20m,True,False,40

   -  The first row is the headers:

      -  ``area`` refers to the ``GeoPackage`` file with target area
         data i.e. a single polygon circle without the extension
         ``.gpkg``.
      -  ``traces`` refers to the ``GeoPackage`` file with trace data
         without the extension ``.gpkg``. The trace dataset has no limit
         on its extent, the analysis is always focused only based on the
         target area.
      -  ``thematic`` refers to the thematic name for a dataset. No
         strict convention for naming.
      -  ``scale`` refers to the scale of observation for the dataset.
         E.g. ``20m`` refers to the drone flight elevation of 20 meters.
         No strict convention for naming. All datasets that are
         subsampled at the same time should be of the same scale.
      -  ``valid`` refers to if the trace/area combo is valid for
         further analysis. Validation is done with ``fractopo``
         ``tracevalidate`` command line tool. You can run validation for
         your datasets with task ``invoke validate-all``. This will
         validate based on the field contents of ``valid``. Content must
         be ``True`` or ``False``. Change it manually after you've
         conducted validation and the traces pass.
      -  ``empty`` is either ``True`` or ``False``. Refers to target
         areas that are devoid of traces but there's really no reason to
         have such areas except for as future work reminder.
      -  ``diameter`` is the diameter of the target area circle.

3. Files and directories:

   -  Value in the ``thematic`` column is the name of the base directory
      where the trace and area data is located. Within the trace and
      area folders the data is in a folder referring to the ``scale``
      column value. E.g.

      .. code:: bash

         ahvenanmaa/
         ├── areas
         │   └── 20m
         │       ├── Flato_20m_1_area.gpkg
         └── traces
             └── 20m
                 ├── Flato_20m_1_traces.gpkg

   -  Either put your trace and target area GeoPackages directly as
      they've been designated in ``relations.csv`` or put them in
      ``unorganized`` folder and run ``invoke organize`` which will
      organize the files automatically using ``relations.csv``.

4. Next up we will perform the steps to get to subsampling your dataset.

   .. code:: bash

      # Conduct fractopo network analysis and save results in points
      # into a GeoPackage
      invoke network-all --points --overwrite

      # Conduct subsampling to generate stage 1 dataset
      # The number designated how many subsamples per target area are
      # made
      invoke network-subsampling --how-many 50

   -  Configure stage 2 subsampling by editing
      ``notebooks/subsampling_config.py`` file. E.g.
      ``circle_names_with_diameter`` variable is the filter for choosing
      which target areas are wanted in stage 2 subsampling.

5. Subsampling is done in jupyter notebooks.

   .. code:: bash

      # Run the following command to start jupyter lab instance
      jupyter lab

   -  Notebook ``Base_Circle_Analysis_Figure_7.ipynb`` needs to be run
      before ``Subsampling_Figures_8_9_and_10.ipynb`` to create base
      circle reference value csv.
   -  Notebooks might require additional configuration to fit your
      datasets.

Post any questions as issues here on GitHub or email me at
nikolasovaskainen@gmail.com.
