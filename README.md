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

### Prerequisites

**1. Install VGGT Module (Recommended)**

The model requires multiview geometry prediction to generate novel views. We recommend installing VGGT:

```shell
git clone https://github.com/facebookresearch/vggt.git
```

> **Note:** VGGT requires additional packages from `requirements_dev.txt`. Install them if you haven't already:
> ```shell
> pip install -r requirements_dev.txt
> ```

### Step-by-Step Inference Guide

#### Step 1: Prepare Your Input Images

1. Create a directory for your reference images (default: `./images`)
2. Place your input images in this directory
3. Supported formats: Standard image formats (`.jpg`, `.png`, etc.)

**Example:**
```
MoAI/
└── images/
    ├── view1.jpg
    ├── view2.jpg
    └── view3.jpg
```

#### Step 2: Configure the Inference Settings

Edit `eval_configs/eval.yaml` to match your setup:

**2.1 Set the Number of Reference Images**

Update `num_viewpoints` to match the number of images in your reference directory:

```yaml
dataset:
  num_viewpoints: 3  # Change this to match your number of input images
```

**2.2 Set the Reference Images Directory**

```yaml
eval_images_dir: "./images"  # Path to your input images
```

#### Step 3: Run Inference with Interactive Camera Control

MoAI provides an **interactive camera positioning system** that lets you manually control the target viewpoint before generating the novel view.

**3.1 Start the Inference Process**

```shell
python inference.py  # Or your main inference script
```

**3.2 Interactive Camera Search**

When prompted, you'll see a preview image (`RENDERING.png`) showing the projected point cloud from the current camera viewpoint.

**Camera Control Commands:**

| Command | Action | Description |
|---------|--------|-------------|
| `W` | Move Forward | Translate camera along z-axis (+0.15 units) |
| `S` | Move Backward | Translate camera along z-axis (-0.15 units) |
| `A` | Move Left | Translate camera along x-axis (-0.15 units) |
| `D` | Move Right | Translate camera along x-axis (+0.15 units) |
| `T` | Pitch Up | Rotate camera around x-axis (+10°) |
| `G` | Pitch Down | Rotate camera around x-axis (-10°) |
| `F` | Yaw Left | Rotate camera around y-axis (+10°) |
| `H` | Yaw Right | Rotate camera around y-axis (-10°) |

**3.3 Interactive Workflow**

1. System displays initial rendering in RENDERING.png
2. Enter camera movement command (e.g., "W" to move forward)
3. System updates RENDERING.png with new viewpoint
4. When prompted "Continue searching, or no?":
   - Type anything to continue adjusting
   - Type "no" to finalize this viewpoint and start generation
5. Repeat steps 2-4 until satisfied with the camera position

**Example Session:**
```
Cmd [W/A/S/D translate, T/F/G/H rotate, END to finish]: W
[RENDERING.png updated]
Continue searching, or no?: yes

Cmd [W/A/S/D translate, T/F/G/H rotate, END to finish]: T
[RENDERING.png updated]
Continue searching, or no?: yes

Cmd [W/A/S/D translate, T/F/G/H rotate, END to finish]: D
[RENDERING.png updated]
Continue searching, or no?: no
[Generation starts...]
```

#### Step 4: View Results

After the generation completes, your output will be saved to the configured output directory. The generated files include:
- Novel view image
- Corresponding geometry/depth map

### Advanced Configuration

#### Fine-Tune Camera Control Sensitivity

You can adjust the camera movement step sizes by editing `main/utils/eval_utils.py`:

```python
def camera_search(cam, cmd, device):
    t_step = 0.15  # Translation step size (smaller = finer control)
    r_step = 10.0  # Rotation step in degrees (smaller = finer control)
    # ...
```

**Recommendations:**
- For precise positioning: `t_step = 0.05`, `r_step = 5.0`
- For quick exploration: `t_step = 0.3`, `r_step = 20.0`

#### Configuration Options

The `eval_configs/eval.yaml` file contains additional settings you can modify:

```yaml
# Normalization settings
normalized_pose: true

# Feature conditioning
use_mesh: true
use_normal: true
use_depthmap: true
use_conf: true

# Model architecture options
use_geo_ref_unet: true
use_warped_img_cond: true
feature_fusion_type: 'warped_feature'

# Noise configuration
noise_offset: 0.0
uncond_ratio: 0.1
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
