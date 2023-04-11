# Instruct-NeRF2NeRF: Editing 3D Scenes with Instructions

This is the official implementation of [Instruct-NeRF2NeRF](https://instruct-nerf2nerf.github.io/).

<video id="teaser" muted autoplay playsinline loop controls width="100%">
    <source id="mp4" src="https://instruct-nerf2nerf.github.io/data/videos/face.mp4" type="video/mp4">
</video>

# Installation

## 1. Install Nerfstudio dependencies

Instruct-NeRF2NeRF is build on Nerfstudio and therefore has the same dependency reqirements. Specfically [PyTorch](https://pytorch.org/) and [tinycudann](https://github.com/NVlabs/tiny-cuda-nn) are required.

Follow the instructions [at this link](https://docs.nerf.studio/en/latest/quickstart/installation.html#dependencies) to create the environment and install dependencies. Only follow the commands up to tinycudann. After the dependencies have been installed, return here.

## 2. Installing Instruct-NeRF2NeRF

Once you have finished installing dependencies, you can install Instruct-NeRF2NeRF using the following command:
```bash
pip install git+https://github.com/ayaanzhaque/instruct-nerf2nerf
```

_Optional_: If you would like to work with the code directly, clone then install the repo:
```bash
git clone https://github.com/ayaanzhaque/instruct-nerf2nerf.git
cd instruct-nerf2nerf
pip install --upgrade pip setuptools
pip install -e .
```

## 3. Checking the install

The following command should include `in2n` as one of the options:
```bash
ns-train -h
```

# Using Instruct-NeRF2NeRF

<video id="pipeline" muted autoplay playsinline loop controls width="100%">
    <source id="mp4" src="https://instruct-nerf2nerf.github.io/data/videos/pipeline_animation.mp4" type="video/mp4">
</video>

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

After the NeRF is trained, you can render the NeRF using the standard Nerfstudio workflow, found [here](https://docs.nerf.studio/en/latest/quickstart/viewer_quickstart.html).

## Training Notes

Our method uses ~16K rays and LPIPS, but not all GPUs have enough memory to run this configuration. As a result, we have provided two alternative configurations which use less memory, but be aware that these configurations lead to decreased performance. The differences are the precision used for IntructPix2Pix and whether LPIPS is used (which requires 4x more rays). The details of each config is provided in the table below.

| Method | Precision of InstructPix2Pix | LPIPS? | Memory |
| ---------------------------------------------------------------------------------------------------- | -------------- | ----------------------------------------------------------------- | ----------------------- |
| ```in2n-big``` | Full | Yes | <15GB |
| ```in2n``` | Half | Yes | <12GB |
| ```in2n-lite``` | Half | No | <10GB |

Please note that training the NeRF on images with resolution larger than 512 will likely cause InstructPix2Pix to throw OOM errors. You can either downscale your dataset yourself and update your ```transforms.json``` file (scale down w, h, fl_x, fl_y, cx, cy), or you can use a smaller image scale provided by Nerfstudio. You can add ```nerfstudio-data --downscale-factor {2,4,6,8}``` to the end of your ```ns-train``` commands.

We recommend capturing data using images from Polycam, as smaller datasets work better and faster with our method.

If you have multiple GPUs, training can be sped up by placing InstructPix2Pix on a separate GPU. To do so, add ```--pipeline.ip2p-device cuda:{device-number}``` to your training command.

# Extending Instruct-NeRF2NeRF

### Issues
Please open Github issues for any installation/usage problems you run into. We've tried to support as broad a range of GPUs as possible, but it might be necessary to provide even more low-footprint versions. Please contribute with any changes to improve memory usage!

### Code structure
To build off Instruct-NeRF2NeRF, we provide explanations of the core code components.

```data/in2n_datamanager.py```: This file is almost identical to the ```base_datamanager.py``` in Nerfstudio. The main difference is that the entire dataset tensor is pre-computed in the ```setup_train``` method as opposed to being sampled in the ```next_train``` method each time.

```in2n_pipeline.py```: This file builds on the pipeline module in Nerfstudio. The ```get_train_loss_dict``` method samples images and places edited images back into the dataset.

```ip2p.py```: This file houses the InstructPix2Pix model (using the ```diffusers``` implementation). The ```edit_image``` method is where an image is denoised using the diffusion model, and a variety of helper methods are contained in this file as well.

```in2n.py```: We overwrite the ```get_loss_dict``` method to use LPIPs loss and L1Loss.

# Citation

You can find our paper on [arXiv](https://arxiv.org/abs/2303.12789).

If you find this code or find the paper useful for your research, please consider citing:

```
@article{instructnerf2023,
    author = {Haque, Ayaan and Tancik, Matthew and Efros, Alexei and Holynski, Aleksander and Kanazawa, Angjoo},
    title = {Instruct-NeRF2NeRF: Editing 3D Scenes with Instructions},
    booktitle = {arXiv preprint 2303.12789},
    year = {2023},
} 
```