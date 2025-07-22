#!/bin/bash

# Exit immediately on error
set -e

# Clone the repo
git clone https://github.com/hieultph/isaac-skypilot.git

# Download and install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
~/miniconda3/bin/conda init bash
conda init bash
source ~/.bashrc

# Change to project directory (check correct path)
cd isaac-skypilot

# Create and activate conda environment
conda create -n gr00t python=3.10
conda activate gr00t

# Install Python dependencies
pip install --upgrade pip setuptools
pip install -e .[base]
pip install --no-build-isolation flash-attn==2.7.1.post4
