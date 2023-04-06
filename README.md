# Instruct-NeRF2NeRF: Editing 3D Scenes with Instructions

This is the official implementation of [Instruct-NeRF2NeRF](https://instruct-nerf2nerf.github.io/).

# Installation

Instruct-NeRF2NeRF follows the integration guidelines described [here](https://docs.nerf.studio/en/latest/developer_guides/new_methods.html) for custom methods within Nerfstudio.

Follow the instructions [at this link](https://github.com/nerfstudio-project/nerfstudio/blob/main/README.md#1-installation-setup-the-environment) to create the environment and install dependencies. After the dependencies have been installed, return here. The instructions have been copied here for convenience.

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

Once you have finished installing dependencies, you can install Instruct-NeRF2NeRF using the following commands:

```bash
git clone https://github.com/ayaanzhaque/instruct-nerf2nerf.git
cd instruct-nerf2nerf
pip install --upgrade pip setuptools
pip install -e .
```

### Checking the install
Run ```ns-train -h```: you should see a list of "subcommands" with in2n.

# Using Instruct-NeRF2NeRF
To edit a NeRF, you must first train a regular ```nerfacto``` scene using your data. To process your custom data, please refer to [this](https://docs.nerf.studio/en/latest/quickstart/custom_dataset.html) documentation.

Once you have your custom data, you can train your initial NeRF with the following command:

```bash
ns-train nerfacto --data {PROCESSED_DATA_DIR}
```

For more details on training a NeRF, see [Nerfstudio documentation](https://docs.nerf.studio/en/latest/quickstart/first_nerf.html).

Once you have fully trained your scene, the checkpoints will be saved to the ```outputs``` directory. Copy the path to the ```nerfstudio_models``` folder.

To start training for editing the NeRF, run the following command:

```bash
ns-train in2n --data {PROCESSED_DATA_DIR} --load-dir {outputs/.../nerfstudio_models} --pipeline.prompt {"prompt"} --pipeline.guidance-scale 7.5 --pipeline.image-guidance-scale 1.5
```

The ```{PROCESSED_DATA_DIR}``` must be the same path as used in training the original NeRF. Using the CLI commands, you can choose the prompt and the guidance scales used for InstructPix2Pix.

**Important**: Please note that training the NeRF on images with resolution than 512 will cause InstructPix2Pix to throw OOM errors. You can either downscale your dataset yourself and update your ```transforms.json``` file (scale down w, h, fl_x, fl_y, cx, cy), or you can use a smaller image scale provided by Nerfstudio. You can add ```nerfstudio-data --downscale-factor {2,4,6,8}``` to the end of your ```ns-train``` commands.

After the NeRF is trained, you can render the NeRF using the standard Nerfstudio workflow, found [here](https://docs.nerf.studio/en/latest/quickstart/viewer_quickstart.html).

## Training Notes

If you have multiple GPUs, training can be sped up by placing InstructPix2Pix on a separate GPU. To do so, add ```--pipeline.ip2p-device cuda:{device-number}``` to your training command.

In our work, we find that using an LPIPS loss can lead to notable improvements in performance. However, using LPIPs requires training with 4x the standard amount of rays in order to sample a useful amount of patches. This will increase the memory requirements. If you would like to train with LPIPs, you can run the following command:

```bash
ns-train in2n --data {PROCESSED_DATA_DIR} --load-dir {outputs/.../nerfstudio_models} --pipeline.prompt {"prompt"} --pipeline.guidance-scale 7.5 --pipeline.image-guidance-scale 1.5 --pipeline.model.use-lpips --pipeline.datamanager.patch-size 32
```
