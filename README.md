<h1 align="center">BeSplat – Gaussian Splatting from a Single Blurry Image and Event Stream</h1>
<p align="center">
    <a href="https://akawincent.github.io">Wenpu Li</a><sup>1,5*</sup> &emsp;&emsp;
    <a href="https://github.com/pianwan">Pian Wan </a><sup>1,2*</sup> &emsp;&emsp;
    <a href="https://wangpeng000.github.io">Peng Wang</a><sup>1,3*</sup> &emsp;&emsp;
    <a href="https://jinghangli.github.io/">Jinghang Li</a><sup>4</sup> &emsp;&emsp;
    <a href="https://sites.google.com/view/zhouyi-joey/home">Yi Zhou</a><sup>4</sup> &emsp;&emsp;
    <a href="https://ethliup.github.io/">Peidong Liu</a><sup>1†</sup>
</p>

<p align="center">
    <sup>*</sup>equal contribution &emsp;&emsp; <sup>†</sup> denotes corresponding author.
</p>

<p align="center">
    <sup>1</sup>Westlake University &emsp;&emsp;
    <sup>2</sup>EPFL &emsp;&emsp;
    <sup>3</sup>Zhejiang University &emsp;&emsp;
    <sup>4</sup>Hunan University &emsp;&emsp; </br>
    <sup>5</sup>Guangdong University of Technology 
</p>

<hr>

<h5 align="center"> This paper was accepted by European Conference on Computer Vision (ECCV) 2024.</h5>
<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub.</h5>

<h5 align="center">

[![arXiv](https://img.shields.io/badge/Arxiv-2407.02174-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2407.02174v3)
[![pdf](https://img.shields.io/badge/PDF-Paper-orange.svg?logo=GoogleDocs)](./doc/2024_ECCV_BeNeRF_camera_ready_paper.pdf) 
[![pdf](https://img.shields.io/badge/PDF-Supplementary-orange.svg?logo=GoogleDocs)](./doc/2024_ECCV_BeNeRF_camera_ready_supplementary.pdf) 
[![pdf](https://img.shields.io/badge/PDF-Poster-orange.svg?logo=GoogleDocs)](https://akawincent.github.io/BeNeRF/demo/Poster.pdf) 
[![Home Page](https://img.shields.io/badge/GitHubPages-ProjectPage-blue.svg?logo=GitHubPages)](https://akawincent.github.io/BeNeRF/)
[![Paper With Code](https://img.shields.io/badge/Website-PaperwithCode-yellow.svg?logo=paperswithcode)](https://paperswithcode.com/paper/benerf-neural-radiance-fields-from-a-single)  
[![Dataset](https://img.shields.io/badge/OneDrive-Dataset-green.svg?logo=ProtonDrive)](https://westlakeu-my.sharepoint.com/:f:/g/personal/cvgl_westlake_edu_cn/EjZNs8MwoXBDqT61v_j5V3EBIoKb8dG9KlYtYmLxcNJG_Q?e=AFXeUB)
![GitHub Repo stars](https://img.shields.io/github/stars/WU-CVGL/BeNeRF)

</h5>

> We explore the possibility of recovering sharp radiance fields (Gaussian splats) and camera motion trajectory from a single motion-blurred image. This allows BeSplat to decode the underlying sharp scene representation and video from a single blurred image and its corresponding event stream.


## 🚀 Updates  
- Project homepage is now live! Check it out [here](https://akawincent.github.io/BeNeRF/).  
- Training, testing, and evaluation codes, along with datasets, are now available.  
- Our paper has been officially accepted to ECCV 2024—congratulations to all collaborators!  


## 🔍 Approach

<p align="center">
    <img src="./doc/Pipeline.png" alt="Pipeline" style="width:75%; height:auto;">
</p>

<div>
Given a single motion-blurred image and its corresponding event stream, BeSplat recovers the underlying 3D scene representation (Gaussian splats) and the camera motion trajectory jointly. Specifically, we represent the 3D scene using Gaussian Splatting and model the camera motion trajectory with a Bézier curve in SE(3) space. Both the blurry image and the accumulated events over a time interval can be synthesized from the 3D scene representation using the estimated camera poses. The scene representation and camera trajectory are optimized by minimizing the discrepancy between the synthesized data and the real-world measurements.
</div>


## 🛠️ Setup Instructions

### Installation

Follow the setup instructions for **3D Gaussian Splatting** for environment requirements and setup. 


### Download Datasets
We use real-world datasets from **E2NeRF**, captured using the DAVIS346 color event camera, and synthetic datasets from **BeNeRF** for evaluations.

- The **real-world datasets** contain five scenes: *letter*, *lego*, *camera*, *plant*, and *toys*.
- The **synthetic datasets** from BeNeRF include three sequences from Unreal Engine: *livingroom*, *whiteroom*, and *pinkcastle*, and two sequences from Blender: *tanabata* and *outdoorpool*.

You can download the datasets from the following links:

- **[Download BeNeRF Datasets](https://westlakeu-my.sharepoint.com/:f:/g/personal/cvgl_westlake_edu_cn/EjZNs8MwoXBDqT61v_j5V3EBIoKb8dG9KlYtYmLxcNJG_Q?e=AFXeUB)**

### Training

```shell
python train.py -s <path to dataset> --eval --deblur # Train with train/test split
```

Additional Command Line Arguments for `train.py`

* `blur_sample_num`: number of key frames for trajectory time sampling
* `deblur`: switch the deblur mode
* `mode`: models of camera motion trajectory (i.e. Linear, Spline, Bezier)
* `bezier_order`: order of the Bézier curve when use Bézier curve for trajectory modeling


### Evaluation

```shell
python train.py -s <path to dataset> --eval # Train with train/test split
python render.py -m <path to trained model> # Generate renderings
python metrics.py -m <path to trained model> # Compute error metrics on renderings
```

Additional Command Line Arguments for `render.py`

* `optim_pose`: optimize the camera pose to align with the dataset

### Render Video

```shell
python render_video.py -m <path to trained model>
```

### Results
You can check our results at the following link.

- [https://arxiv.org/abs/2407.02174](https://arxiv.org/abs/2407.02174)
- [https://akawincent.github.io/BeNeRF/](https://akawincent.github.io/BeNeRF/)

## ✒️ Citation 
If you find this repository useful, please consider citing our paper:
```bibtex
@inproceedings{li2024benerf,
    author = {Wenpu Li and Pian Wan and Peng Wang and Jinghang Li and Yi Zhou and Peidong Liu},
    title = {BeNeRF: Neural Radiance Fields from a Single Blurry Image and Event Stream},
    booktitle = {European Conference on Computer Vision (ECCV)},
    year = {2024}
} 
```

## 🙏 Acknowledgment
In our work, the camera trajectory optimization was inspired by [Deblur-GS](https://github.com/google-research/deblur-gs), and the event stream integration into Gaussian Splatting was inspired by the methodology used in [BeNeRF](https://github.com/akawincent/BeNeRF). The overall code framework is based on both repositories. We appreciate the efforts of the contributors to these amazing projects.

