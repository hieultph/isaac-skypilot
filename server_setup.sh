#!/bin/bash
set -e

# Define persistent directory
PERSIST_DIR="/workspace/setup"
mkdir -p "$PERSIST_DIR"

# 1. Install Miniconda into /workspace
if [ ! -d "/workspace/miniconda3" ]; then
    echo "Installing Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "$PERSIST_DIR/miniconda.sh"
    bash "$PERSIST_DIR/miniconda.sh" -b -p /workspace/miniconda3
fi

# 2. Set up conda
eval "$(/workspace/miniconda3/bin/conda shell.bash hook)"

# Ensure conda envs are saved inside /workspace
cat <<EOF > /workspace/.condarc
envs_dirs:
  - /workspace/conda_envs
pkgs_dirs:
  - /workspace/conda_pkgs
EOF

export CONDARC=/workspace/.condarc

# Add source to bashrc (custom .bashrc in /workspace)
if ! grep -q "source /workspace/.bashrc" ~/.bashrc 2>/dev/null; then
    echo "source /workspace/.bashrc" >> ~/.bashrc
fi

# Write .bashrc to load conda every time
cat <<EOF > /workspace/.bashrc
# Custom .bashrc for RunPod
source /workspace/miniconda3/etc/profile.d/conda.sh
conda activate gr00t
EOF

# 3. Create Conda env (persisted)
if [ ! -d "/workspace/conda_envs/gr00t" ]; then
    conda create -p /workspace/conda_envs/gr00t python=3.10
    conda config --append envs_dirs /workspace/conda_envs
fi
conda activate gr00t

# 6. Python dependencies
pip install --upgrade pip setuptools
pip install -e .[base]
pip install --no-build-isolation flash-attn==2.7.1.post4

echo "✅ Setup complete and persistent in /workspace."
