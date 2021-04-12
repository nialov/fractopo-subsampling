#!/usr/bin/env bash

echo "Cloning fractopo-subsampling repository."

# Clone the fractopo-subsampling repository
git clone https://github.com/nialov/fractopo-subsampling --depth 1

echo "Copying scripts, requirements.txt and notebooks from the cloned
directory."

# Copy scripts and notebooks from the cloned fractopo-subsampling directory
cp fractopo-subsampling/scripts_and_notebooks/* --target-directory .
