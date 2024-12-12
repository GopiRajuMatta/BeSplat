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
    <img src="./doc/pipeline.png" alt="Pipeline" style="width:75%; height:auto;">
</p>

<div>
Given a single motion-blurred image and its corresponding event stream, BeSplat recovers the underlying 3D scene representation (Gaussian splats) and the camera motion trajectory jointly. Specifically, we represent the 3D scene using Gaussian Splatting and model the camera motion trajectory with a Bézier curve in SE(3) space. Both the blurry image and the accumulated events over a time interval can be synthesized from the 3D scene representation using the estimated camera poses. The scene representation and camera trajectory are optimized by minimizing the discrepancy between the synthesized data and the real-world measurements.
</div>


## 🔥 QuickStart
### 1. Installation
In the path where your want to store code, enter the following terminal command:

```bash
git clone https://github.com/WU-CVGL/BeNeRF.git
cd BeNeRF
conda create -n benerf python=3.9
conda activate benerf
pip install -r requirements.txt
```
If the network speed is slow when using pip to download dependencies, you may consider changing the pip source:
```bash
python -m pip install --upgrade pip
pip config set global.index-url https://mirrors.bfsu.edu.cn/pypi/web/simple
```

### 2. Download Datasets
You can download `BeNeRF_Datasets` by clicking this [link](https://westlakeu-my.sharepoint.com/:f:/g/personal/cvgl_westlake_edu_cn/EjZNs8MwoXBDqT61v_j5V3EBIoKb8dG9KlYtYmLxcNJG_Q?e=AFXeUB). 

It contains real-world dataset(e.g. *e2nerf_real*) and synthetic dataset(*benerf_blender*, *benerf_unreal* and *e2nerf_synthetic*). For the every scene, the folder `images` includes blurry images and folder `events` includes event stream for entire sequence. The timestamps of the start and end of exposure for each image are stored in a `txt` file. These timestamps are used to segment the entire event stream into individual event streams corresponding to each image. Additionally, we provide grounth sharp images in folder `images_test` for syntehtic dataset in order to evaluate metrics.

For the scenes of `benerf_blender` and `benerf_unreal`, We provide two versions of the images: one in color and one in grayscale. The grayscale version is used for quantitative evaluation, while the color version is used for qualitative evaluation. Since event-enhanced baseline methods we compared to can only run with gray image due to the single channel event stream we synthesized, we compute the metrics of all methods with gray images for consistency.

### 3. Train
First, you need to modify the path of datasets in config file:
```yml
datadir = XXXXX/BeNeRF_Datasets/real or synthetic/<dataset>/<scene>
```
Then, You can train model by entering terminal command as follow:
```bash
python train.py --device <cuda_id> --config ./configs/<dataset>/<scene>.txt --index <img_id>
```
We use wandb as a viewr to moniter the training process by defalut:

<p align="center">
    <img src="./doc/viewer.png" alt="display loss" style="width:75%; height:auto;">
</p>

After training, all results including render image, render video, camera trajectory and checkpoint file will be saved in the path specified by `logdir/<img_id>/` in the config file.

### 4. Test
You can test the model by loading the checkpoint file saved in the `logdir/<img_id>/` path. We provide three options to test model.

#### Extract poses
```bash
python test.py --device <cuda_id> --config ./configs/<dataset>/<scene>.txt --index <img_id> \
               --extract_poses --num_extract_poses <the number of poses you want to extract>
```
We have set the default number of extracted poses to 19, i.e., `num_extract_poses = 19`.

#### Render images
```bash
python test.py --device <cuda_id> --config ./configs/<dataset>/<scene>.txt --index <img_id> \ 
               --render_images --num_render_images <the number of images you want to render>
```
We have set the default number of render images to 19, i.e., `num_render_images = 19`.

#### Render video
```bash
python test.py --device <cuda_id> --config ./configs/<dataset>/<scene>.txt --index <img_id> --render_video
```

Of course, you can choose all three options simultaneously to test the model.
```bash
python test.py --device <cuda_id> --config ./configs/<dataset>/<scene>.txt --index <img_id> \
               --extract_poses --num_extract_poses <the number of poses you want to extract> \
               --render_images --num_render_images <the number of images you want to render> \
               --render_video                                                           
```
The test results will be saved in `logdir/<img_id>/test_results` path.


### 5. Evaluation
To compute metrics like PSNR, SSIM, LPIPS for evaluating model performance on synthetic dataset, you can run script as follow:
```bash
python evaluate.py --dataset <dataset name> --scene <scene name> \
                   --result <path to folder of image results> --groundtruth <path to folder of groundtruth image>
```
For real-world datasets, since sharp ground truth images are not available, we choose to use the no-reference image quality metric `BRISQUE` to quantitatively evaluate the model's performance on real-world datasets. You can run the following script in MATLAB.
```bash
matlab -r "eval_brisque('<path to folder of image results>')"

```
## Results
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
In our work, the camera optimization and event stream integration into NeRF were inspired by [BAD-NeRF](https://github.com/WU-CVGL/BAD-NeRF) and [E-NeRF](https://github.com/knelk/enerf), respectively. The overall code framework is based on [nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch/). We appreciate the effort of the contributors to these amazing repositories.
