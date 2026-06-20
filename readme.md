# Code Usage Guide(EN)


## News

**2026-05-22**: This paper has been accepted by **Communications Physics-Nature**.

arxiv preprint link: https://arxiv.org/abs/2605.25679


## Overview

This repository provides the **full codebase** and **demo datasets** used in the paper:

**“Transformer-based Neural Operators for 3D Wind Field Prediction over Complex Mountainous Terrain”**


## Abstract
Accurate prediction of three-dimensional wind fields over complex mountainous terrain is essential for renewable energy deployment and regional weather modeling. Existing computational fluid dynamics simulations are accurate but slow and require laborious mesh generation, while many deep learning methods struggle to capture sharp local wind variations induced by complex terrain. Here, we present a transformer-based dual-attention neural-operator framework for rapid wind-field prediction over complex mountainous terrain. Trained on a large simulation dataset spanning diverse terrain geometries and inflow conditions, the framework enables rapid prediction of steady-state wind field while maintaining competitive accuracy. It transfers robustly to real-world mountainous sites, reducing relative error by about 10\% compared with existing neural-operator baselines. With sparse observations covering only 1\% of the domain, prediction error is further reduced by 16.89\% relative to the model without observations and by 32.75\% relative to advanced baselines on unseen terrains. This framework provides a practical tool for wind-resource assessment over complex mountainous terrain and atmosphere–surface interaction studies.

## Dataset

The demo dataset can be downloaded from：link: https://huggingface.co/datasets/yujiaA/Demo-dataset-of-complex-mountainous-terrain 

You can directly replace the folders 'all_dataset_1' and 'all_dataset_zero_shot' with the download folders

Please organize the dataset in the same format as all_vtk_organize.  The required directory structure is shown below:

if you are the first time to use, the dataset will be handled automatically to the numpy format.

The vtk files should be organized as follows:

```txt
├── vtk_files
│   └── chatou-1
│       └── result
│           └── 0.0
│               └── VTK
│                   ├── bottom
│                   │   └── bottom_cutted.vtk
│                   └── cutted.vtk
```

Note: 0.0 is the inlet angle. bottom_cutted.vtk is the bottom surface, and cutted.vtk is the entire computational domain.

The numpy files will be saved in the following directory:

```txt
├── np_files
│   └── chatou-1
│       └── result
│           └── 0.0
│               └── VTK
│                   ├── x.npy
│                   ├── y.npy
```

Note 1: The x.npy is the input data, and y.npy is the output data.


Note 2: The zero-shot dataset are also organized in the same format, see the folder: all_dataset_zero_shot


Note 3: Although the full dataset is unavailable because of the agreement limitation, we provide a partial dataset for training and testing.

## Environment Configuration

Taken Patchsolver as example:

1. you should switch to the Patchsolver directory:

```bash
cd Patchsolver
```

2. create a new environment for python 3.10:

3. install the torch, torchvision and pytorch_geometric

install [torch torchvision](https://pytorch.org/)

install [pytorch_geometric](https://github.com/pyg-team/pytorch_geometric).


4. install the required packages:

```bash
pip install -r requirements.txt
```

or

pip install scipy matplotlib pyyaml einops timm vtk xdem scikit-learn


## Train model


1. run the train on the train dataset, infer on the test dataset, infer on the zero-shot dataset:


```bash
bash patchsolver.sh
```


3. run the case with sparse data (train, test on the test dataset and on the zero-shot dataset):


```bash

bash sparse_patchsolver_run.sh

```


## run time and run memory

1. The memory usage of the model is about 30-60G.

2. The running time of the model is about 4 days.




## For the other baseline models

1. For the transolver model, you only need to modify the model name (cfd_model) in the config file to the Transolver.

The implementation of the Transolver model is based on the following open-source projects: https://github.com/thuml/Transolver


2. For the PatchGTO and AeroGTO model, you should switch to the corresponding directory and execute the corresponding command.

```bash
# PatchGTO
python main_dp.py --config ./config/MSE_times3.json

python infer.py --config ./config/MSE_times3.json
```

and

```sh
# AeroGTO
python main_dp.py --config ./config/MSE_times3.json

python infer.py --config ./config/MSE_times3.json
```


The implementation of the AeroGTO model is based on the following open-source projects: https://github.com/pengwei07/AeroGTO


3. For the other baselines, you can follow the following links:

GNOT : https://github.com/HaoZhongkai/GNOT

GINO: https://github.com/NeuralOperator/neuraloperator






