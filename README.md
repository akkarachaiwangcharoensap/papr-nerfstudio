# Nerfstudio Installation Guide on WSL (Windows 11)

This guide provides a concise overview of installing [Nerfstudio](https://github.com/nerfstudio-project/nerfstudio) and its dependencies using **Conda** on **WSL (Windows Subsystem for Linux)** running on **Windows 11**.

> Adapted from:
> - https://github.com/nerfstudio-project/nerfstudio/
> - https://docs.nerf.studio/quickstart/installation.html
> - https://gist.github.com/kauffmanes/5e74916617f9993bc3479f401dfec7da

---

## Quick Overview

1. [Install Conda](#install-conda--create-conda-env)  
2. [Install Nerfstudio & Dependencies](#install-nerfstudio-and-its-dependencies)  
3. [Train & View Models](#train--view-models)  
4. [Troubleshooting](#troubleshooting)

---

## Install Conda & Create Conda Env

### Download the Installer

1. Create a shell script, e.g., `conda_installer.sh`, and make it executable:

```bash
chmod +x conda_installer.sh
```

2. Download the installer (replace `[CONDA_VERSION]` as needed):

```bash
wget https://repo.continuum.io/archive/[CONDA_VERSION]
```

Example:

```bash
wget https://repo.continuum.io/archive/Anaconda3-2024.10-1-Linux-x86_64.sh
```

> 🔗 Full list of versions: [Anaconda Archive](https://repo.continuum.io/archive)

### Create Conda Environment

```bash
conda create --name papr-nerfstudio -y python=3.10
conda activate papr-nerfstudio
pip install --upgrade pip
```

---

## Install Nerfstudio and Its Dependencies

### 1. Prepare Your Environment

```bash
pip uninstall torch torchvision functorch tinycudann
```

### 2. Install PyTorch with CUDA 12.8

```bash
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
conda install -c "nvidia/label/cuda-12.8.0" cuda-toolkit
```

### 3. Install `tiny-cuda-nn`

```bash
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

### 4. Clone Nerfstudio

```bash
git clone https://github.com/nerfstudio-project/nerfstudio.git --branch v1.1.5
# OR your custom fork:
# git clone git@github.com:akkarachaiwangcharoensap/papr-nerfstudio.git

cd nerfstudio
pip install --upgrade pip setuptools
pip install -e .
```

---

## Train & View Models

### Download Sample Data

```bash
ns-download-data nerfstudio --capture-name=person
```

> 🔗 Full list available in [download_data.py](https://github.com/nerfstudio-project/nerfstudio/blob/f31f3bba12841955102f3f3846ee9f855f4a6878/nerfstudio/scripts/downloads/download_data.py#L115-L142).

### Train the Model

```bash
ns-train nerfacto --data data/nerfstudio/person
```

### Resume Training

```bash
ns-train nerfacto --data data/nerfstudio/person --load-dir "outputs/.../nerfstudio_models"
```

### Launch Viewer

```bash
ns-viewer --load-config "outputs/.../config.yml"
```

---

## Troubleshooting

### Increase File Descriptor Limit

```bash
ulimit -n 65535
```

### `libcuda.so not found` Error

```bash
ln -s /usr/lib/wsl/lib/libcuda.so $CONDA_PREFIX/lib/
ln -s /usr/lib/wsl/lib/libcuda.so.1 $CONDA_PREFIX/lib/
```

### `nvcc` Not Found

```bash
which nvcc
rm -rf <path-to-nvcc>
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch --no-cache-dir
```

### `weights_only` Warning

Set this environment variable before training or launching viewer:

```bash
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
```

---
