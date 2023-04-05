# Instruct-NeRF2NeRF: Editing 3D Scenes with Instructions

This is the official implementation of [Instruct-NeRF2NeRF](https://instruct-nerf2nerf.github.io/).

# Installation

Instruct-NeRF2NeRF follows the integration guidelines described [here](https://docs.nerf.studio/en/latest/developer_guides/new_methods.html) for custom methods within Nerfstudio.

## 1. Installation: Setup the environment

### Prerequisites

You must have an NVIDIA video card with CUDA installed on the system. This library has been tested with version 11.3 of CUDA. You can find more information about installing CUDA [here](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html)

### Create environment

Nerfstudio requires `python >= 3.7`. We recommend using conda to manage dependencies. Make sure to install [Conda](https://docs.conda.io/en/latest/miniconda.html) before proceeding.

```bash
conda create --name in2n -y python=3.8
conda activate in2n
python -m pip install --upgrade pip
```

### Dependencies

Install pytorch with CUDA (this repo has been tested with CUDA 11.3 and CUDA 11.7) and [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn)

```bash
pip install torch==1.13.1 torchvision functorch --extra-index-url https://download.pytorch.org/whl/cu117
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

See [Dependencies](https://github.com/nerfstudio-project/nerfstudio/blob/main/docs/quickstart/installation.md#dependencies)
in the Installation documentation for more.

### Installing Instruct-NeRF2NeRf
```bash
git clone https://github.com/ayaanzhaque/instruct-nerf2nerf.git
cd instruct-nerf2nerf
pip install --upgrade pip setuptools
pip install -e .
```

Checking the install
Run ```ns-train -h```: you should see a list of "subcommands" with in2n.

# Using Instruct-NeRF2NeRF
To edit a NeRF, you must first train a regular ```nerfacto``` scene using your data. To do so, run ```ns-train nerfacto --data {PROCESSED_DATA_DIR}```. For more details on training a NeRF, see [Nerfstudio documentation](https://docs.nerf.studio/en/latest/quickstart/first_nerf.html).

Once you have your scene, run the following command to train en edited version for your NeRF.
```bash
ns-train in2n --data {PROCESSED_DATA_DIR} --load-dir {outputs/.../nerfstudio_models} --pipeline.prompt {"prompt"} --pipeline.guidance-scale 7.5 --pipeline.image-guidance-scale 1.5 --pipeline.image-guidance-scale 1.5
```

Using the CLI commands, you can choose the prompt and the guidance scales used for InstructPix2Pix.