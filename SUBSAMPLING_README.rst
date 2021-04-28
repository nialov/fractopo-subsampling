How to recreate subsampling environment
=======================================

Note! Only for Linux-based systems.

1.  Create working directory e.g. ~/projects/subsampling

2.  Switch to the working directory.

3.  Go to
    https://www.kaggle.com/nasurz/getaberget-fracture-trace-dataset and
    download the fracture trace dataset.

    -  Alternative to manual downloading is the ``kaggle-api`` Python
       app. If you have it installed (``pip install kaggle``) and
       configured
       (https://github.com/Kaggle/kaggle-api#api-credentials), you can
       run:

       .. code:: bash

          kaggle datasets download nasurz/getaberget-fracture-trace-dataset

    -  Either way you will have downloaded a zip file with the dataset.

4.  Extract files and directory within the zip to the working directory.

5.  Create a Python virtual environment.

    -  You may use pip, pipenv or poetry.
    -  Python must be 3.8.

6.  Activate the virtual environment.

7.  Download some prerequisites for installing
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

**If you do not wish to conduct subsampling or base circle
characterization, skip to the next step (10.).**

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
notebooks:

.. code:: bash

   invoke network-all --overwrite --notebooks

To store results in a single GeoPackage as points for spatial analysis
and reference value plotting:

.. code:: bash

   invoke network-all --overwrite --points

To conduct stage 1 subsampling for 25 times for each base circle:

.. code:: bash

   invoke network-subsampling --how-many 25

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
       subsampling results as csvs from kaggle as well.

    -  Or alternatively repeat the subsampling or base circle
       characterization that is introduced in the previous step to get
       unique subsamples.

    -  Configure notebook analysis in
       ``notebooks/subsampling_config.py`` and within the notebooks
       themselves.

11. If you wish to use the exact datasets of stage 1 and 2 subsampling
    that I used, continue to 12 to download them. Otherwise, we're done!
    See step 9 for brief introduction to stage 1 subsampling and base circle
    characterization.
