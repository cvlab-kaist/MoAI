# Aligned Novel View Image and Geometry Synthesis via Cross-modal Attention Instillation

[![Project Site](https://img.shields.io/badge/Project-Web-green)](https://cvlab-kaist.github.io/MoAI/) &nbsp;
[![Github](https://img.shields.io/badge/Github-Repo-orange?logo=github)](https://github.com/cvlab-kaist/MoAI) &nbsp; 
[![arXiv](https://img.shields.io/badge/arXiv-2506.11924-red?logo=arxiv)](https://arxiv.org/abs/2506.11924) &nbsp; 
<!-- [![Spaces](https://img.shields.io/badge/Spaces-Demo-yellow?logo=huggingface)]() &nbsp;  -->
<!-- [![Models](https://img.shields.io/badge/Models-checkpoints-blue?logo=huggingface)]() &nbsp;  -->

[Introduction](#introduction)
| [Demo](#demo)
| [Examples](#examples)
| [How to use](#how-to-use)
| [Citation](#citation)
| [Acknowledgements](#acknowledgements)

![concept image](https://github.com/cvlab-kaist/MoAI/blob/main/MoAI/assets/teaser.png)

## Introduction

This repository is an official implementation for the paper "[Aligned Novel View Image and Geometry Synthesis via Cross-modal Attention Instillation](https://cvlab-kaist.github.io/MoAI/)". We introduce a diffusion-based framework that performs aligned novel view image and geometry generation via a warping‐and‐inpainting methodology. For detailed information, please refer to the [paper](https://arxiv.org/abs/2506.11924).

![Framework](https://github.com/cvlab-kaist/MoAI/blob/main/MoAI/assets/architecture.png)

## Overview and Examples

Our model can generate novel view image and geometry in extrapolative, far-away camera viewpoints from arbitrary number of unposed reference images. This is enabled by our cross-**Mo**dal **A**ttention **I**istillation (**MoAI**), in which the spatial attention maps of image generation pipeline is instilled into the geometry generation pipeline during training and inference for synergyistic effects. 

You can find examples on our [project page](https://cvlab-kaist.github.io/MoAI/) and on our [paper](https://arxiv.org/abs/2506.11924). 

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
    Download our checkpoints from our MoAI [Huggingface Hub](https://huggingface.co/minseop-kwak/moai-checkpoints).

2. Pretrained models:
    - [sd-image-variations-diffusers](https://huggingface.co/lambdalabs/sd-image-variations-diffusers)
      - download `image_encoder/config.json` and `image_encoder/pytorch_model.bin` to `checkpoints/image_encoder`

The final `checkpoints` directory must look like this:

```
MoAI
└── checkpoints
    ├── image_encoder
    │   ├── config.json
    │   └── pytorch_model.bin
    ├── configs
    │   ├── image_config.json
    │   └── geometry_config.json
    ├── main
    │   ├── denoising_unet.pth
    │   ├── geometry_unet.pth
    │   ├── pose_guider.pth
    │   ├── geo_reference_unet.pth
    │   └── reference_unet.pth
```

### Inference Instructions

#### (Recommended) Install VGGT module

The model requires multiview geometry prediction to generate novel views. To this end, users can install one of multiview geometry prediction models publicly available. We used and recommend VGGT.
``` shell
git clone https://github.com/facebookresearch/vggt.git
```

To use VGGT, please install `requirements_dev.txt` for additional packages.

#### Configuration Setup

Before running inference, configure the following parameters in `eval_configs/eval.yaml`:

**1. Dataset Configuration**

Specify the number of reference images (should match the number of images at the reference images directory):
```yaml
dataset:
  num_viewpoints: 1
```

- `num_viewpoints`: Specifies the number of input reference images

**2. Reference Images Directory**

Specify the directory containing your reference images:
```yaml
eval_images_dir: "./images"
```

Place your input images in this directory before running inference.

**3. Interactive Target Camera Control**

MoAI provides an interactive camera search tool that allows you to manually control the target camera viewpoint. The system projects the reference image's point cloud into the target view, rendering a preview that updates in real-time as you adjust the camera pose.

During this process, a preview image (`RENDERING.png`) is saved after each adjustment, showing the projected point cloud from the current camera viewpoint. This allows you to interactively find the desired novel view before running the full generation pipeline.

**Camera Control Commands (Optional)**

You can adjust the `t_step` (translation) and `r_step` (rotation) values in the code below (located in `main/utils/eval_utils.py`) for more fine-grained or coarser camera control.

```python
def camera_search(cam, cmd, device):
    """
    Adjusts camera pose based on user input commands.
    
    Translation commands (step size: 0.15):
    - W/S: Move forward/backward along z-axis
    - A/D: Move left/right along x-axis
    
    Rotation commands (step size: 10°):
    - T/G: Pitch up/down (rotation around x-axis)
    - F/H: Yaw left/right (rotation around y-axis)
    """
    t_step = 0.15  # Translation step size
    r_step = 10.0  # Rotation step in degrees

    if cmd == 'W':
        T = make_translation_matrix(0, 0, t_step, device)
        cam = T @ cam
    elif cmd == 'S':
        T = make_translation_matrix(0, 0, -t_step, device)
        cam = T @ cam
    # ... [additional movement commands]
    
    return cam[None,...]
```

## Citation

``` bibtex
@misc{kwak2025moai,
  title={Aligned Novel View Image and Geometry Synthesis via Cross-modal Attention Instillation}, 
  author={Min-Seop Kwak and Junho Kim and Sangdoo Yun and Dongyoon Han and Taekyoung Kim and Seungryong Kim and Jin-Hwa Kim},
  year={2025},
  eprint={2506.11924},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2506.11924}, 
}
```

## Acknowledgements

Our codes are based on [Moore-AnimateAnyone](https://github.com/MooreThreads/Moore-AnimateAnyone) and other repositories it is based on. We thank the authors of relevant repositories and papers.
