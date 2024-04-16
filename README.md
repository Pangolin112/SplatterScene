# splatter-image
Official implementation of **"Splatter Image: Ultra-Fast Single-View 3D Reconstruction" (CVPR 2024)**

[16 Apr 2024] Several big updates to the project since the first release:
- We can now reconstruct **any object**: we trained open-category model trained on Objaverse in just 7 GPU days
- We now have a [demo](https://huggingface.co/spaces/szymanowiczs/splatter_image) where you can upload your own pictures of **any** object and have our model reconstruct it
- Models for all 6 datasets are now !released!. We trained 6 models: on Objaverse, multi-category ShapeNet, CO3D hydrants, CO3D teddybears, ShapeNet cars and ShapeNet chairs.
- SOTA on multi-category ShapeNet
- Support for multi-GPU training
- No camera pose pre-processing in CO3D

<video controls muted loop autoplay preload playsinline width="75%">
  <source src="./demo_examples/gso_loop.mp4" type="video/mp4">
</video>

# Demo

Check out the online [demo](https://huggingface.co/spaces/szymanowiczs/splatter_image). Running the demo locally will often be even faster and you will be able to see the loops rendered with Gaussian Splatting (as opposed to the extracted .ply object which can show artefacts). To run the demo locally, simply follow the installation instructions below, and afterwards call:
```
python gradio_app.py
```

# Installation

1. Create a conda environment: 
```
conda create --name splatter-image
conda activate splatter-image
```

Install Pytorch following [official instructions](https://pytorch.org). Pytorch / Python / Pytorch3D combination that was verified to work is:
- Python 3.8, Pytorch 1.13.0, CUDA 11.6, Pytorch3D 0.7.2
Alternatively, you can create a separate environment with Pytorch3D 0.7.2, which you use just for CO3D data preprocessing. Then, once CO3D had been preprocessed, you can use these combinations of Python / Pytorch too. 
- Python 3.7, Pytorch 1.12.1, CUDA 11.6
- Python 3.8, Pytorch 2.1.1, CUDA 12.1

Install other requirements:
```
pip install -r requirements.txt
```

2. Install Gaussian Splatting renderer, i.e. the library for rendering a Gaussian Point cloud to an image. To do so, pull the [Gaussian Splatting repository](https://github.com/graphdeco-inria/gaussian-splatting/tree/main) and, with your conda environment activated, run `pip install submodules/diff-gaussian-rasterization`. You will need to meet the [hardware and software requirements](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/README.md#hardware-requirements). We did all our experimentation on an NVIDIA A6000 GPU and speed measurements on an NVIDIA V100 GPU. 

3. If you want to train on CO3D data you will need to install Pytorch3D 0.7.2. See instructions [here](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md). It is recommended to install with pip from a pre-built binary. Find a compatible binary [here](https://anaconda.org/pytorch3d/pytorch3d/files?page=5) and install it with `pip`. For example, with Python 3.8, Pytorch 1.13.0, CUDA 11.6 run
`pip install --no-index --no-cache-dir pytorch3d -f https://anaconda.org/pytorch3d/pytorch3d/0.7.2/download/linux-64/pytorch3d-0.7.2-py38_cu116_pyt1130.tar.bz2`.

# Data

## ShapeNet cars and chairs
For training / evaluating on ShapeNet-SRN classes (cars, chairs) please download the srn_\*.zip (\* = cars or chairs) from [PixelNeRF data folder](https://drive.google.com/drive/folders/1PsT3uKwqHHD2bEEHkIXB99AlIjtmrEiR?usp=sharing). Unzip the data file and change `SHAPENET_DATASET_ROOT` in `datasets/srn.py` to the parent folder of the unzipped folder. For example, if your folder structure is: `/home/user/SRN/srn_cars/cars_train`, in `datasets/srn.py` set  `SHAPENET_DATASET_ROOT="/home/user/SRN"`. No additional preprocessing is needed.

## CO3D hydrants and teddybears
For training / evaluating on CO3D download the hydrant and teddybear classes from the [CO3D release](https://github.com/facebookresearch/co3d/tree/main). To do so, run the following commands:
```
git clone https://github.com/facebookresearch/co3d.git
cd co3d
mkdir DOWNLOAD_FOLDER
python ./co3d/download_dataset.py --download_folder DOWNLOAD_FOLDER --download_categories hydrant,teddybear
```
Next, set `CO3D_RAW_ROOT` to your `DOWNLOAD_FOLDER` in `data_preprocessing/preoprocess_co3d.py`. Set `CO3D_OUT_ROOT` to where you want to store preprocessed data. Run 
```
python -m data_preprocessing.preprocess_co3d
``` 
and set `CO3D_DATASET_ROOT:=CO3D_OUT_ROOT`.

## Multi-category ShapeNet
For multi-category ShapeNet we use the ShapeNet 64x64 dataset by NMR hosted by DVR authors which can be downloaded [here](https://s3.eu-central-1.amazonaws.com/avg-projects/differentiable_volumetric_rendering/data/NMR_Dataset.zip).
Unzip the folder and set `NMR_DATASET_ROOT` to the directory that holds sub-category folders after unzipping. In other words, `NMR_DATASET_ROOT` directory should contain folders `02691156`, `02828884`, `02933112` etc.

## Objaverse

For training on Objaverse we used renderings from Zero-1-to-3 which can be downloaded with the follownig command:
```
wget https://tri-ml-public.s3.amazonaws.com/datasets/views_release.tar.gz
```
Disclaimer: note that the renderings are generated with Objaverse. The renderings as a whole are released under the ODC-By 1.0 license. The licenses for the renderings of individual objects are released under the same license creative commons that they are in Objaverse.

Additionally, please download `lvis-annotations-filtered.json` from the [model repository](). 
This json which holds the list of IDs of objects from the LVIS subset. These assets are of higher quality.

Set `OBJAVERSE_ROOT` in `datasets/objaverse.py` to the directory of the unzipped folder with renderings, and set `OBJAVERSE_LVIS_ANNOTATION_PATH` in the same file to the directory of the downloaded `.json` file.

Note that Objaverse dataset is meant for training and validation only. It does not have a test subset.

## Google Scanned Objects

For evaluating the model trained on Objaverse we use Google Scanned Objects dataset to ensure no overlap with the training set. Download [renderings provided by Free3D](`https://drive.google.com/file/d/1tV-qpiD5e-GzrjW5dQpTRviZa4YV326b/view`). Unzip the downloaded folder and set `GSO_ROOT` in `datasets/gso.py` to the directory of the unzipped folder.

Note that Google Scanned Objects dataset is not meant for training. It is used to test the model trained on Objaverse.

# Using this repository

## Pretrained models

Pretrained models for all datasets are now available via [Huggingface Models](https://huggingface.co/szymanowiczs/splatter-image-v1). If you just want to run qualitative / quantitative evaluation, do don't need to dowload them manually, they will be used automatically if you run the evaluation script (see below).

You can also download them manually if you wish to do so, by manually clicking the download button on the [Huggingface model files page](https://huggingface.co/szymanowiczs/splatter-image-v1). Download the config file with it and see `eval.py` for how the model is loaded.


## Evaluation

Once you downloaded the relevant dataset, evaluation can be run with 
```
python eval.py $dataset_name
```
`$dataset_name` is the name of the dataset. We support:
- `gso` (Google Scanned Objects), 
- `objaverse` (Objaverse-LVIS), 
- `nmr` (multi-category ShapeNet), 
- `hydrants` (CO3D hydrants), 
- `teddybears` (CO3D teddybears), 
- `cars` (ShapeNet cars), 
- `chairs` (ShapeNet chairs).
The code will automatically download the relevant model for the requested dataset.

You can also train your own models and evaluate it with 
```
python eval.py $dataset_name --experiment_path $experiment_path
```
`$experiment_path` should hold a `model_latest.pth` file and a `.hydra` folder with `config.yaml` inside it.

To evaluate on the validation split, call with option `--split val`.

To save renders of the objects with the camera moving in a loop, call with option `--split vis`. With this option the quantitative scores are not returned since ground truth images are not available in all datasets.

You can set for how many objects to save renders with option `--save_vis`.
You can set where to save the renders with option `--out_folder`.

## Training

Single-view models are trained in two stages, first without LPIPS (most of the training), followed by fine-tuning with LPIPS.
1. The first stage is ran with:
      ```
      python train_network.py +dataset=$dataset_name
      ```
      where $dataset_name is one of [cars,chairs,hydrants,teddybears,nmr,objaverse].
      Once it is completed, place the output directory path in configs/experiment/lpips_$experiment_name.yaml in the option `opt.pretrained_ckpt` (by default set to null).
2. Run second stage with:
      ```
      python train_network.py +dataset=$dataset_name +experiment=$lpips_experiment_name
      ```
      Which `$lpips_experiment_name` to use depends on the dataset.
      If $dataset_name is in [cars,hydrants,teddybears], use lpips_100k.yaml.
      If $dataset_name is chairs, use lpips_200k.yaml.
      If $dataset_name is nmr, use lpips_nmr.yaml.
      If $dataset_name is objaverse, use lpips_objaverse.yaml.
      Remember to place the directory of the model from the first stage in the appropriate .yaml file before launching the second stage.

To train a 2-view model run:
```
python train_network.py +dataset=cars cam_embd=pose_pos data.input_images=2 opt.imgs_per_obj=5
```

## Code structure

Training loop is implemented in `train_network.py` and evaluation code is in `eval.py`. Datasets are implemented in `datasets/srn.py` and `datasets/co3d.py`. Model is implemented in `scene/gaussian_predictor.py`. The call to renderer can be found in `gaussian_renderer/__init__.py`.

## Camera conventions

Gaussian rasterizer assumes row-major order of rigid body transform matrices, i.e. that position vectors are row vectors. It also requires cameras in the COLMAP / OpenCV convention, i.e., that x points right, y down, and z away from the camera (forward).

# BibTeX

```
@inproceedings{szymanowicz24splatter,
      title={Splatter Image: Ultra-Fast Single-View 3D Reconstruction},
      author={Stanislaw Szymanowicz and Christian Rupprecht and Andrea Vedaldi},
      year={2024},
      booktitle={The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
}
```

# Acknowledgements

S. Szymanowicz is supported by an EPSRC Doctoral Training Partnerships Scholarship (DTP) EP/R513295/1 and the Oxford-Ashton Scholarship.
A. Vedaldi is supported by ERC-CoG UNION 101001212.
We thank Eldar Insafutdinov for his help with installation requirements.
