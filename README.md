# Aligned Novel View Image and Geometry Synthesis via Cross-modal Attention Instillation

[![Project Site](https://img.shields.io/badge/Project-Web-green)](https://cvlab-kaist.github.io/MoAI/) &nbsp;
[![Github](https://img.shields.io/badge/Github-Repo-orange?logo=github)](https://github.com/cvlab-kaist/MoAI) &nbsp; 
<!-- [![Spaces](https://img.shields.io/badge/Spaces-Demo-yellow?logo=huggingface)]() &nbsp;  -->
<!-- [![Models](https://img.shields.io/badge/Models-checkpoints-blue?logo=huggingface)]() &nbsp;  -->
<!-- [![arXiv](https://img.shields.io/badge/arXiv-2405.17251-red?logo=arxiv)](https://arxiv.org/abs/2405.17251) -->

[Introduction](#introduction)
| [Demo](#demo)
| [Examples](#examples)
| [How to use](#how-to-use)
| [Citation](#citation)
| [Acknowledgements](#acknowledgements)

![concept image](https://github.com/cvlab-kaist/MoAI/blob/main/MoAI/assets/teaser.png)

## Introduction

This repository is an official implementation for the paper "[Aligned Novel View Image and Geometry Synthesis via Cross-modal Attention Instillation](https://cvlab-kaist.github.io/MoAI/)". We introduce a diffusion-based framework that performs aligned novel view image and geometry generation via a warping‐and‐inpainting methodology. For detailed information, please refer to the [paper](https://arxiv.org/abs/2405.17251).

![Framework](https://github.com/cvlab-kaist/MoAI/blob/main/MoAI/assets/architecture.png)

## Examples

Our model can handle images from various domains including indoor/outdoor scenes, and even illustrations with challenging camera viewpoint changes.

You can find examples on our [project page](https://cvlab-kaist.github.io/MoAI/) and on our [paper](https://arxiv.org/abs/2405.17251). 

![Examples](https://github.com/cvlab-kaist/MoAI/blob/main/MoAI/assets/Qual_image.png)

<!-- Generated novel views can be used for 3D reconstruction. In the example below, we reconstructed a 3D scene via [InstantSplat](https://instantsplat.github.io/). We generated the video using [this implementation](https://github.com/ONground-Korea/unofficial-Instantsplat).

<video autoplay loop src="https://github.com/user-attachments/assets/b3362776-815c-426f-bf39-d04722eb8a6f" width="852" height="480"></video> -->

## How to use

### Environment

We tested our codes on Ubuntu 20.04 with nVidia A6000 GPU. If you're using other machines like Windows, consider using Docker. You can either add packages to your python environment or use Docker to build an python environment. Commands below are all expected to run in the root directory of the repository.

#### Add dependencies to your python environment

We tested the environment with python `>=3.10` and CUDA `=11.8`. To add mandatory dependencies run the command below.

``` shell
pip install -r requirements.txt
```

To run developmental codes such as the example provided in jupyter notebook and the live demo implemented by gradio, add extra dependencies via the command below.

``` shell
pip install -r requirements_dev.txt
```

#### Weight download script

``` shell
./scripts/download_weights.sh ./checkpoints
```

#### Manual download

> [!NOTE]
> Models and checkpoints provided below may be distributed under different licenses. Users are required to check licenses carefully on their behalf.

1. Our finetuned models:
    Download of finetuned models will soon be available.

2. Pretrained models:
    - [sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse)
      - download `config.json` and `diffusion_pytorch_model.safetensors` to `checkpoints/sd-vae-ft-mse`
    - [sd-image-variations-diffusers](https://huggingface.co/lambdalabs/sd-image-variations-diffusers)
      - download `image_encoder/config.json` and `image_encoder/pytorch_model.bin` to `checkpoints/image_encoder`

The final `checkpoints` directory must look like this:

```
MoAI
└── checkpoints
    ├── image_encoder
    │   ├── config.json
    │   └── pytorch_model.bin
    ├── main
    │   ├── config.json
    │   ├── geometry_config.json
    │   ├── denoising_unet.pth
    │   ├── geometry_unet.pth
    │   ├── pose_guider.pth
    │   ├── geo_reference_unet.pth
    │   └── reference_unet.pth
    └── sd-vae-ft-mse
        ├── config.json
        └── diffusion_pytorch_model.safetensors
```

### Inference

#### (Recommended) Install VGGT module

The model requires multiview geometry prediction to generate novel views. To this end, users can install one of multiview geometry prediction models publicly available. We used and recommend VGGT.

``` shell
git clone https://github.com/facebookresearch/vggt.git
```

To use VGGT, please install `requirements_dev.txt` for additional packages.

<!-- #### API

**Input Preparation**

Load the input image and estimate the corresponding depth map. Create camera matrices for the intrinsic and extrinsic parameters. [ops.py](genwarp/ops.py) has helper functions to create matrices.

``` python
from PIL import Image
from torchvision.transforms.functional import to_tensor

src_image = to_tensor(Image.open(image_file).convert('RGB'))[None].cuda()
src_depth = depth_estimator.infer(src_image)
```

``` python
import torch
from ops import camera_lookat, get_projection_matrix

proj_mtx = get_projection_matrix(
    fovy=fovy,
    aspect_wh=1.,
    near=near,
    far=far
)

src_view_mtx = camera_lookat(
    torch.tensor([[0., 0., 0.]]),  # From (0, 0, 0)
    torch.tensor([[-1., 0., 0.]]), # Cast rays to -x
    torch.tensor([[0., 0., 1.]])   # z-up
)

tar_view_mtx = camera_lookat(
    torch.tensor([[-0.1, 2., 1.]]), # Camera eye position
    torch.tensor([[-5., 0., 0.]]),  # Looking at.
    z_up  # z-up
)

rel_view_mtx = (
    tar_view_mtx @ torch.linalg.inv(src_view_mtx.float())
).to(src_image)
``` -->

## Citation

``` bibtex
@misc{kwak2025moai,
  title={Cross-modal Attention Instillation for Aligned Novel View Image and Geometry Synthesis}, 
  author={Min-Seop Kwak and Junho Kim and Sangdoo Yun and Dongyoon Han and Taekyoung Kim and Seungryong Kim and Jin-Hwa Kim},
  year={2025},
  eprint={2506.UNDECIDED},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2506.UNDECIDED}, 
}
```

## Acknowledgements

Our codes are based on [Moore-AnimateAnyone](https://github.com/MooreThreads/Moore-AnimateAnyone) and other repositories it is based on. We thank the authors of relevant repositories and papers.
