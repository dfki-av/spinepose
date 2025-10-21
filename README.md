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

- [Inference](#inference)  
- [SpineTrack Dataset](#spinetrack-dataset)  
- [Results and Evaluation](#results-and-evaluation) 

If you use our models or dataset, please cite our work as described in the [Citation](#citation) section.

---

## Inference
**Recommended Python Version:** 3.9–3.12

For quick spinal keypoint estimation, we release optimized ONNX models via the `spinepose` package on PyPI:

```bash
pip install spinepose
```

On Linux/Windows with CUDA available, install the GPU version:

```bash
pip install spinepose[gpu]
```

### Using the CLI

```
usage: spinepose [-h] (--version | --input_path INPUT_PATH) [--vis-path VIS_PATH] [--save-path SAVE_PATH] [--mode {xlarge,large,medium,small}] [--nosmooth] [--spine-only]

SpinePose Inference

options:
  -h, --help            show this help message and exit
  --version, -V         Print the version and exit.
  --input_path INPUT_PATH, -i INPUT_PATH
                        Path to the input image or video
  --vis-path VIS_PATH, -o VIS_PATH
                        Path to save the output image or video
  --save-path SAVE_PATH, -s SAVE_PATH
                        Save predictions in OpenPose format (.json for image or folder for video).
  --mode {xlarge,large,medium,small}, -m {xlarge,large,medium,small}
                        Model size. Choose from: xlarge, large, medium, small (default: medium)
  --nosmooth            Disable keypoint smoothing for video inference (default: enabled)
  --spine-only          Only use 9 spine keypoints (default: use all 37 keypoints)
```

For example, to run inference on a video and save only spine keypoints in OpenPose format:

```bash
spinepose --input_path path/to/video.mp4 --save-path output_path.json --spine-only
```

This automatically downloads the model weights (if not already present) and outputs the annotated image or video. Use spinepose -h to view all available options, including GPU usage and confidence thresholds.

### Using the Python API

```python
import cv2
from spinepose import SpinePoseEstimator

# Initialize estimator (downloads ONNX model if not found locally)
estimator = SpinePoseEstimator(device='cuda')

# Perform inference on a single image
image = cv2.imread('path/to/image.jpg')
keypoints, scores = estimator(image)
visualized = estimator.visualize(image, keypoints, scores)
cv2.imwrite('output.jpg', visualized)
```

Or, for a simplified interface:

```python
from spinepose.inference import infer_image, infer_video

# Single image inference
results = infer_image('path/to/image.jpg', vis_path='output.jpg')

# Video inference with optional temporal smoothing
results = infer_video('path/to/video.mp4', vis_path='output_video.mp4', use_smoothing=True)
```

## SpineTrack Dataset

SpineTrack is available on [HuggingFace](https://doi.org/10.57967/hf/5114). The dataset comprises:

- **SpineTrack-Real**
  A collection of real-world images annotated with nine spinal keypoints in addition to standard body joints. An active learning pipeline, combining pretrained neural annotators and human corrections, refines keypoints across diverse poses.

- **SpineTrack-Unreal**
  A synthetic subset rendered using Unreal Engine, paired with precise ground-truth from a biomechanically aligned OpenSim model. These synthetic images facilitate pretraining and complement real-world data.

To download:

```bash
git lfs install
git clone https://huggingface.co/datasets/saifkhichi96/spinetrack
```

Alternatively, use `wget` to download the dataset directly:

```bash
wget https://huggingface.co/datasets/saifkhichi96/spinetrack/resolve/main/annotations.zip
wget https://huggingface.co/datasets/saifkhichi96/spinetrack/resolve/main/images.zip
```

In both cases, the dataset will download two zipped folders: `annotations` (24.8 MB) and `images` (19.4 GB), which can be unzipped to obtain the following structure:

```plaintext
spinetrack
├── annotations/
│   ├── person_keypoints_train-real-coco.json
│   ├── person_keypoints_train-real-yoga.json
│   ├── person_keypoints_train-unreal.json
│   └── person_keypoints_val2017.json
└── images/
    ├── train-real-coco/
    ├── train-real-yoga/
    ├── train-unreal/
    └── val2017/
```

All annotations are in COCO format and can be used with standard pose estimation libraries.

## Results and Evaluation

We benchmark SpinePose against state-of-the-art lightweight pose estimation methods on COCO, Halpe, and our SpineTrack dataset. The results are summarized below, with SpinePose models highlighted in gray. Only 26 body keypoints are used for Halpe evaluations.

<table border="1" cellspacing="0" cellpadding="6" style="border-collapse:collapse; text-align:center; font-family:Arial; font-size:13px;">
  <thead style="background-color:#f0f0f0; font-weight:bold;">
    <tr>
      <th>Method</th>
      <th>Train Data</th>
      <th>Kpts</th>
      <th colspan="2">COCO</th>
      <th colspan="2">Halpe26</th>
      <th colspan="2">Body</th>
      <th colspan="2">Feet</th>
      <th colspan="2">Spine</th>
      <th colspan="2">Overall</th>
      <th>Params (M)</th>
      <th>FLOPs (G)</th>
    </tr>
    <tr>
      <th></th><th></th><th></th>
      <th>AP</th><th>AR</th>
      <th>AP</th><th>AR</th>
      <th>AP</th><th>AR</th>
      <th>AP</th><th>AR</th>
      <th>AP</th><th>AR</th>
      <th>AP</th><th>AR</th>
      <th></th><th></th>
    </tr>
  </thead>
  <tbody>
    <tr><td>SimCC-MBV2</td><td>COCO</td><td>17</td><td>62.0</td><td>67.8</td><td>33.2</td><td>43.9</td><td>72.1</td><td>75.6</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.1</td><td>0.1</td><td>2.29</td><td>0.31</td></tr>
    <tr><td>RTMPose-t</td><td>Body8</td><td>26</td><td>65.9</td><td>71.3</td><td>68.0</td><td>73.2</td><td>76.9</td><td>80.0</td><td>74.1</td><td>79.7</td><td>0.0</td><td>0.0</td><td>15.8</td><td>17.9</td><td>3.51</td><td>0.37</td></tr>
    <tr><td>RTMPose-s</td><td>Body8</td><td>26</td><td>69.7</td><td>74.7</td><td>72.0</td><td>76.7</td><td>80.9</td><td>83.6</td><td>78.9</td><td>83.5</td><td>0.0</td><td>0.0</td><td>17.2</td><td>19.4</td><td>5.70</td><td>0.70</td></tr>
    <tr style="background-color:#e6e6e6; font-weight:bold;"><td>SpinePose-s</td><td>SpineTrack</td><td>37</td><td>68.2</td><td>73.1</td><td>70.6</td><td>75.2</td><td>79.1</td><td>82.1</td><td>77.5</td><td>82.9</td><td>89.6</td><td>90.7</td><td>84.2</td><td>86.2</td><td>5.98</td><td>0.72</td></tr>
    <tr><td colspan="17" style="background-color:#d0d0d0; height:3px;"></td></tr>
    <tr><td>SimCC-ViPNAS</td><td>COCO</td><td>17</td><td>69.5</td><td>75.5</td><td>36.9</td><td>49.7</td><td>79.6</td><td>83.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.2</td><td>0.2</td><td>8.65</td><td>0.80</td></tr>
    <tr><td>RTMPose-m</td><td>Body8</td><td>26</td><td>75.1</td><td>80.0</td><td>76.7</td><td>81.3</td><td>85.5</td><td>87.9</td><td>84.1</td><td>88.2</td><td>0.0</td><td>0.0</td><td>19.4</td><td>21.4</td><td>13.93</td><td>1.95</td></tr>
    <tr style="background-color:#e6e6e6; font-weight:bold;"><td>SpinePose-m</td><td>SpineTrack</td><td>37</td><td>73.0</td><td>77.5</td><td>75.0</td><td>79.2</td><td>84.0</td><td>86.4</td><td>83.5</td><td>87.4</td><td>91.4</td><td>92.5</td><td>88.0</td><td>89.5</td><td>14.34</td><td>1.98</td></tr>
    <tr><td colspan="17" style="background-color:#d0d0d0; height:3px;"></td></tr>
    <tr><td>RTMPose-l</td><td>Body8</td><td>26</td><td>76.9</td><td>81.5</td><td>78.4</td><td>82.9</td><td>86.8</td><td>89.2</td><td>86.9</td><td>90.0</td><td>0.0</td><td>0.0</td><td>20.0</td><td>22.0</td><td>28.11</td><td>4.19</td></tr>
    <tr><td>RTMW-m</td><td>Cocktail14</td><td>133</td><td>73.8</td><td>78.7</td><td>63.8</td><td>68.5</td><td>84.3</td><td>86.7</td><td>83.0</td><td>87.2</td><td>0.0</td><td>0.0</td><td>6.2</td><td>7.6</td><td>32.26</td><td>4.31</td></tr>
    <tr><td>SimCC-ResNet50</td><td>COCO</td><td>17</td><td>72.1</td><td>78.2</td><td>38.7</td><td>51.6</td><td>81.8</td><td>85.2</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.2</td><td>0.2</td><td>36.75</td><td>5.50</td></tr>
    <tr style="background-color:#e6e6e6; font-weight:bold;"><td>SpinePose-l</td><td>SpineTrack</td><td>37</td><td>75.2</td><td>79.5</td><td>77.0</td><td>81.1</td><td>85.4</td><td>87.7</td><td>85.5</td><td>89.2</td><td>91.0</td><td>92.2</td><td>88.4</td><td>90.0</td><td>28.66</td><td>4.22</td></tr>
    <tr><td colspan="17" style="background-color:#d0d0d0; height:3px;"></td></tr>
    <tr><td>SimCC-ResNet50*</td><td>COCO</td><td>17</td><td>73.4</td><td>79.0</td><td>39.8</td><td>52.4</td><td>83.2</td><td>86.2</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.3</td><td>0.3</td><td>43.29</td><td>12.42</td></tr>
    <tr><td>RTMPose-x*</td><td>Body8</td><td>26</td><td>78.8</td><td>83.4</td><td>80.0</td><td>84.4</td><td>88.6</td><td>90.6</td><td>88.4</td><td>91.4</td><td>0.0</td><td>0.0</td><td>21.0</td><td>22.9</td><td>50.00</td><td>17.29</td></tr>
    <tr><td>RTMW-l*</td><td>Cocktail14</td><td>133</td><td>75.6</td><td>80.4</td><td>65.4</td><td>70.1</td><td>86.0</td><td>88.3</td><td>85.6</td><td>89.2</td><td>0.0</td><td>0.0</td><td>8.1</td><td>8.1</td><td>57.20</td><td>7.91</td></tr>
    <tr><td>RTMW-l*</td><td>Cocktail14</td><td>133</td><td>77.2</td><td>82.3</td><td>66.6</td><td>71.8</td><td>87.3</td><td>89.9</td><td>88.3</td><td>91.3</td><td>0.0</td><td>0.0</td><td>8.6</td><td>8.6</td><td>57.35</td><td>17.69</td></tr>
    <tr style="background-color:#e6e6e6; font-weight:bold;"><td>SpinePose-x*</td><td>SpineTrack</td><td>37</td><td>75.9</td><td>80.1</td><td>77.6</td><td>81.8</td><td>86.3</td><td>88.5</td><td>86.3</td><td>89.7</td><td>89.3</td><td>91.0</td><td>88.9</td><td>89.9</td><td>50.69</td><td>17.37</td></tr>
  </tbody>
</table>

For evaluation instructions and to reproduce the results reported in our paper, please refer to the `evaluation` branch of this repository:

```bash
git clone https://github.com/dfki-av/spinepose.git
cd spinepose
git checkout evaluation
```

The [README in the evaluation branch](https://github.com/dfki-av/spinepose/blob/evaluation/README.md) provides detailed steps for setting up the evaluation environment and running the evaluation scripts on the SpineTrack dataset.

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
