## Introduction

Instance segmentation-based approach to shrub delineation. Apply transfer learning using Mask R-CNN model to obtain shrub masks over RGB images.

## Data

The example tiles to apply the model were collected over a burnt forest (2003) near Encinacorba (Spain), July 17th 2024. They have a spatial resolution of 5 centimeters.

The `data` folder contains the `logs/mask_rcnn_juniper.h5` file, which includes the weights of the trained model using the dataset created by [Khaldi et al. (2023)](https://zenodo.org/records/10058715).

## Installation

### Manual installation using conda

First, make sure you have conda installed from any of the official sources ([Download Miniconda](https://docs.conda.io/en/latest/miniconda.html)).

Go to the project directory:

```bash
cd /d/shrubs-segmentation
```

Install and set up the environment with the correct dependencies. Run the following line:

```bash
conda create -n mrcnn_tf2 -c conda-forge -y python=3.7
```

> Note: Python 3.7 is an stable version used to test the implementation.

> Note: Using [mamba](https://github.com/conda-forge/miniforge) will shorten the installation times.

Then, activate the environment and install the dependencies using `pip`:

```bash
conda activate mrcnn_tf2
python -m pip install -r requirements.txt
```

If the environment were installed correctly, the jupyter notebooks containing the operations to apply and validate the segmentation can be executed. However, the georeferenced masks used in the current project need geopandas and rasterio libraries. In order to avoid version conflicts with the packages used in the `mrcnn` module, a new environment is set with the following line:

```bash
conda env create -f environment.yml -y
```

Inside the last environment, the codes `Python\clip_img.py` and `Python\convert_masks.py` can be executed. The first one splits the main orthomosaic into smaller tiles, and the second one transform the spatial files containing the shrub masks into a plain json file in order to retrieve them to train the Mask RCNN model ???????????????.

## Usage

First, download the current repository.

```bash
git clone https://github.com/
```

The `bin` folder contains the `mrcnn` module from [Ahmed Gad's Mask-RCNN-TF2 repository](https://github.com/ahmedfgad/Mask-RCNN-TF2). It is used to apply the model Mask R-CNN to the dataset.

> The original [Matterport Mask RCNN repo](https://github.com/matterport/Mask_RCNN) is built with TensorFlow 1.15 which is now no longer supported and it has been updated by TensorFlow 2.

### Re-train the model

The first step is downloading the original JuniperMapper dataset, and store it inside `data` folder.

The second step is convert the annotations from JuniperMapper dataset into a valid Mask RCNN format. Use the following code:

```bash
python convert_annotations.py
```

Then, use the `Notebooks/shrubs_train.ipynb` to retrain the model.

### Apply the model

We trained a Mask RCNN model with the JuniperMapper dataset, using the Photo Interpreted images. We tried to replicate model hyperparams from [Khaldi et al. 2024](https://doi.org/10.1016/j.jag.2024.104191) paper, but they used a Detectron2 instance and our workflow follows an older version of the Mask RCNN model. This was due to incopatibilities between Detectron2 and Windows OS. Our hyperparameters are:

| Hyperparameter | Best value |
| -------------- | ---------- |
| Batch size | 8 |
| Initial learning rate | 0.001 |
| Data augmentation | No augmentation |
| Feature extractor (backbone) | ResNet101 |

Note: We use the learning rate proposed by the [authors of the original Mask RCNN repo](https://github.com/matterport/Mask_RCNN). Using higher values often causes the weights to explode, especially when using a small batch size.

> It might be related to differences between how Caffe and TensorFlow compute gradients (sum vs mean across batches and GPUs). Or, maybe the official model uses gradient clipping to avoid this issue. We do use gradient clipping, but don't set it too aggressively. We found that smaller learning rates converge faster anyway so we go with that.

Model trained with above hyperparameters in 1560 minutes, using a machine with:

| Component name | Model |
| -------------- | ----- |
| Processor | Intel(R) Core(TM) i7-14700KF (3.40 GHz) |
| RAM instalada | 64 GB |
| Graphic card | NVIDIA GeForce RTX 5070 Ti (16 GB) |
| OS | Windows 11 x64 |

## Custom `shrub` classess

*Describe the custom classess to handle the data to classify and validate bin/shrubs/utils.py*

## Funding

