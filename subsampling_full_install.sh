#!/usr/bin/env bash


# Note! 
#   - Requires kaggle cli tool installer and configured.
#   - Requires virtualenv cli tool installed.
#   - Requires git cli tool installed.
#   - Requires python3 installed.

# Run from a freshly made and EMPTY working directory

echo "Downloading fracture traces and areas from kaggle"

kaggle datasets download nasurz/getaberget-fracture-trace-dataset --unzip

echo "Creating local virtualenv in subsampling_venv with virtualenv"

virtualenv subsampling_venv

echo "Activating local virtualenv"

source subsampling_venv/bin/activate

echo "Running subsampling notebook and script download script"

curl https://raw.githubusercontent.com/nialov/fractopo-subsampling/master/subsampling.sh | bash

echo "Installing pip requirements into virtualenv"

pip install -r requirements.txt

echo "Downloading subsampling GeoPackage and csvs from kaggle"

kaggle datasets download nasurz/getaberget-subsampled-fracture-network-dataset --unzip

echo "Moving downloaded datasets into place"

mkdir results/subsampling/collected -p
mkdir results/subsampling/cached_subsamples -p

mv Ahvenanmaa_analysis_points.gpkg results
mv stage_1_subsampling_results.csv results/subsampling/collected/
mv stage_2_aggregated_subsampling_results.csv results/subsampling/cached_subsamples/

echo "Installation completed."

echo "Changing directory to ./notebooks"

cd notebooks

echo "Running notebooks in ./notebooks to test that everything works"

ipython Base_Circle_Analysis_Figure_7.ipynb
ipython Subsampling_Figures_8_9_and_10.ipynb

cd -

echo "Activate the virtualenv in subsampling_venv and then run jupyter lab to"
echo "start inspecting and running notebooks"

echo "e.g. to activate the virtualenv if you are using bash:"
echo "source subsampling_venv/bin/activate"
