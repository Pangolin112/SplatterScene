# Splatter Scene

<img src="./demo_examples/poster.png"
            alt="Poster."/>

# Installation

1. Create a conda environment: 
```
conda create --name splatter-scene
conda activate splatter-scene
```

Install Pytorch following [official instructions](https://pytorch.org). Pytorch / Python combination that was verified to work is:
- Python 3.8, PyTorch 2.1.1, CUDA 12.4

Install other requirements:
```
pip install -r requirements.txt
```

2. Install Gaussian Splatting renderer, i.e. the library for rendering a Gaussian Point cloud to an image. To do so, pull the [Gaussian Splatting repository](https://github.com/graphdeco-inria/gaussian-splatting/tree/main) and, with your conda environment activated, run `pip install submodules/diff-gaussian-rasterization`. You will need to meet the [hardware and software requirements](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/README.md#hardware-requirements). We did all our experimentation on an NVIDIA RTX4090 GPU. 


# Data

## ScanNet++
For training / evaluating on ScanNet++ please download the dataset from [ScanNet++ homepage](https://kaldir.vc.in.tum.de/scannetpp/). Preprocess the file and change `SCANNETPP_DATASET_ROOT` in `datasets/scannetpp.py` to the parent folder of the dataset folder. For example, if your folder structure is: `/home/user/ScanNet++/scannetpp/scannetpp_train`, in `datasets/scannetpp.py` set  `SCANNETPP_DATASET_ROOT="/home/user/ScanNet++"`. 


# Using this repository

## Pretrained models

Pretrained models for all datasets are now available via [Huggingface Models](https://huggingface.co/szymanowiczs/splatter-image-v1). 

You can also download them manually if you wish to do so, by manually clicking the download button on the [Huggingface model files page](https://huggingface.co/szymanowiczs/splatter-image-v1). Download the config file with it and see `eval.py` for how the model is loaded.


## Evaluation

Once you downloaded the relevant dataset, evaluation can be run with 
```
python eval.py scannetpp
```

You can also train your own models and evaluate it with 
```
python eval.py scannetpp --experiment_path $experiment_path
```
`$experiment_path` should hold a `model_latest.pth` file and a `.hydra` folder with `config.yaml` inside it.

To evaluate on the validation split, call with option `--split val`.

You can set for how many objects to save renders with option `--save_vis`.
You can set where to save the renders with option `--out_folder`.

## Training

Run the training for ScanNet++ with:
```
python train_network.py +dataset=scannetpp cam_embd=pose_pos
```

To train a 2-view model run:
```
python train_network.py +dataset=scannetpp cam_embd=pose_pos data.input_images=2 opt.imgs_per_obj=5
```

## Code structure

Training loop is implemented in `train_network.py` and evaluation code is in `eval.py`. Datasets are implemented in `datasets/ScanNet++.py`. Model is implemented in `scene/gaussian_predictor.py`. The call to renderer can be found in `gaussian_renderer/__init__.py`.

## Camera conventions

Gaussian rasterizer assumes row-major order of rigid body transform matrices, i.e. that position vectors are row vectors. It also requires cameras in the COLMAP / OpenCV convention, i.e., that x points right, y down, and z away from the camera (forward).

# Acknowledgements

This work is the extension of **"Szymanowicz et al. Splatter Image: Ultra-Fast Single-View 3D Reconstruction" (CVPR 2024)**, from object-level method to a scene-level method. We thank the authors of the original paper for their code structure implementation.

We thank Barbara Roessle for her insightful help during the project period.
