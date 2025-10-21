# Towards Unconstrained 2D Pose Estimation of the Human Spine

<div align="center">

[![Home](https://img.shields.io/badge/Project-Homepage-pink.svg)](https://saifkhichi.com/research/spinepose/)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-gold.svg)](https://doi.org/10.57967/hf/5114)
[![Conference](https://img.shields.io/badge/CVPRW-2025-blue.svg)](https://openaccess.thecvf.com/content/CVPR2025W/CVSPORTS/html/Khan_Towards_Unconstrained_2D_Pose_Estimation_of_the_Human_Spine_CVPRW_2025_paper.html)
[![arXiv](https://img.shields.io/badge/arXiv-2504.08110-B31B1B.svg)](https://arxiv.org/abs/2504.08110)
[![PyPI version](https://img.shields.io/pypi/v/spinepose.svg)](https://pypi.org/project/spinepose/)
![PyPI - License](https://img.shields.io/pypi/l/spinepose)

![](data/demo/outputs/video1.gif)
![](data/demo/outputs/video2.gif)
</div>

---

> __Abstract__: _We present SpineTrack, the first comprehensive dataset for 2D spine pose estimation in unconstrained settings, addressing a crucial need in sports analytics, healthcare, and realistic animation. Existing pose datasets often simplify the spine to a single rigid segment, overlooking the nuanced articulation required for accurate motion analysis. In contrast, SpineTrack annotates nine detailed spinal keypoints across two complementary subsets: a synthetic set comprising 25k annotations created using Unreal Engine with biomechanical alignment through OpenSim, and a real-world set comprising over 33k annotations curated via an active learning pipeline that iteratively refines automated annotations with human feedback. This integrated approach ensures anatomically consistent labels at scale, even for challenging, in-the-wild images. We further introduce SpinePose, extending state-of-the-art body pose estimators using knowledge distillation and an anatomical regularization strategy to jointly predict body and spine keypoints. Our experiments in both general and sports-specific contexts validate the effectiveness of SpineTrack for precise spine pose estimation, establishing a robust foundation for future research in advanced biomechanical analysis and 3D spine reconstruction in the wild._

## Overview

Official repository for the CVPR 2025 workshop paper "Towards Unconstrained 2D Pose Estimation of the Human Spine" by Muhammad Saif Ullah Khan, Stephan Krauß, and Didier Stricker. This project provides an easy-to-install Python package, pretrained model checkpoints, the SpineTrack dataset, and evaluation scripts to reproduce our results.

- [Installation & Environment Setup](#installation-and-environment-setup)
- [Preparing the Evaluation Datasets](#preparing-the-evaluation-datasets)
- [Downloading Pretrained Models](#downloading-pretrained-models)
- [Running Evaluation](#running-evaluation)

If you use our models or dataset, please cite our work as described in the [Citation](#citation) section.

---

## Installation and Environment Setup

MMPose, MMDetection, and MMCV are required to run the evaluation scripts. Please follow the instructions on [MMPose](https://mmpose.readthedocs.io/en/latest/installation.html) to get started.

We used Python 3.8 with the following package versions for evaluation:

```plain
mmcv==2.1.0
mmdet==3.2.0
mmengine==0.10.7
mmpose==1.3.2
```

## Preparing the Evaluation Datasets

We use COCO, Halpe, and SpineTrack. Follow MMPose docs for COCO and Halpe:

- COCO: https://mmpose.readthedocs.io/en/latest/dataset_zoo/2d_body_keypoint.html#coco  
- Halpe: https://mmpose.readthedocs.io/en/latest/dataset_zoo/2d_wholebody_keypoint.html#halpe

To get SpineTrack, either clone with Git LFS or download the two archives:

```bash
# Option A: Git LFS (recommended)
git lfs install
git clone https://huggingface.co/datasets/saifkhichi96/spinetrack

# Option B: direct download
wget https://huggingface.co/datasets/saifkhichi96/spinetrack/resolve/main/annotations.zip
wget https://huggingface.co/datasets/saifkhichi96/spinetrack/resolve/main/images.zip
unzip annotations.zip -d data/spinetrack/
unzip images.zip -d data/spinetrack/
```

Expected layout (place under `data/spinetrack/`):

```plaintext
data/
└─ spinetrack/
   ├─ annotations/
   │  └─ person_keypoints_val2017.json
   └─ images/
      └─ val2017/
```

Only the annotations and val2017 images are required for evaluation.

## Downloading Pretrained Models

To download the pretrained SpinePose models, run:

```bash
./scripts/download_models.sh
```

This willl create a `data/checkpoints/spinepose/` directory with the following files:

```plaintext
data/
└─ checkpoints/
   └─ spinepose/
      ├─ spinepose-s_32xb256-10e_spinetrack-256x192.pth
      ├─ spinepose-m_32xb256-10e_spinetrack-256x192.pth
      ├─ spinepose-l_32xb256-10e_spinetrack-256x192.pth
      └─ spinepose-x_32xb128-10e_spinetrack-384x288.pth
```

## Running Evaluation

Run evaluation on SpineTrack using:

```bash
./scripts/evaluate_models.sh
```

This will run evaluation for all four models on the SpineTrack validation set and print the results:

```plain
Model               Body AP   Body AR   Feet AP   Feet AR   Spine AP  Spine AR  
--------------------------------------------------------------------------------
spinepose-s         79.25     82.11     77.45     82.97     89.61     90.76     
spinepose-m         84.00     86.39     83.40     87.37     91.44     92.59     
spinepose-l         85.44     87.68     85.59     89.25     90.95     92.17     
spinepose-x         86.26     88.54     86.28     89.75     89.26     90.98
```

Detailed results will be saved in `work_dirs/` folder. To include COCO and Halpe evaluation, ensure the datasets are properly set up and set `include_coco` and `include_halpe` flags to `True` in [`configs/data_config.py`](configs/data_config.py).

---

## Citation

If this project or dataset proves helpful in your work, please cite:

```bibtex
@InProceedings{Khan_2025_CVPR,
    author    = {Khan, Muhammad Saif Ullah and Krau{\ss}, Stephan and Stricker, Didier},
    title     = {Towards Unconstrained 2D Pose Estimation of the Human Spine},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2025},
    pages     = {6171-6180}
}
```

## License

This project is released under the [CC-BY-NC-4.0 License](LICENSE). Commercial use is prohibited, and appropriate attribution is required for research or educational applications.
