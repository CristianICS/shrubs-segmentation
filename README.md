## Introduction

Instance segmentation-based approach to shrub delineation. Apply transfer learning using Mask R-CNN models to obtain shrub masks over RGB images.

We have used Matterport Mask R-CNN and also the Detectron2 architecture proposed by [Khaldi et al. 2024](https://doi.org/10.1016/j.jag.2024.104191) to make a comparisson.

## Data

The example tiles to apply the model were collected over a burnt forest on 17th of July, and the fire occurred in 2003, near Encinacorba (Spain). Imagery have a spatial resolution of 5 centimeters.

TODO: Check the following sentence. Those weights seem to contain the Khaldi et al. hyperparams reprduced by the Matterport Mask R-CNN architecture. In this case these are deprecated files. Remove these.
<!-- The `data` folder contains the `logs/mask_rcnn_juniper.h5` file, which includes the weights of the trained model using the dataset created by [Khaldi et al. (2023)](https://zenodo.org/records/10058715). -->

<!-- The paper's settings cannot be reproduced exactly through configuration alone. Ahmed Gad's repository implements the Matterport ResNet101-FPN architecture, whereas ResNet101-C4 is a different Mask R-CNN backbone/head design. Selecting BACKBONE = "resnet101" does not convert the repository from FPN to C4. The repository defines FPN backbone strides and a 256-channel top-down pyramid explicitly. -->

The data to do transfer learning is downloading from the original [JuniperMapper dataset](https://zenodo.org/records/10058715), and store it inside `data` folder.

## Matterport Mask R-CNN

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

Inside the last environment, the codes `Python\crop_tiles.py` and `Python\convert_annotations.py` can be executed. The first one splits the main orthomosaic into smaller tiles, and the second one transform the JuniperMapper annotations into a valid format for Matterport's Mask RCNN model.

### Usage

First, download the current repository.

```bash
git clone https://github.com/
```

The `bin` folder contains the `mrcnn` module from [Ahmed Gad's Mask-RCNN-TF2 repository](https://github.com/ahmedfgad/Mask-RCNN-TF2). It is used to apply the model Mask R-CNN to the dataset.

> The original [Matterport Mask RCNN repo](https://github.com/matterport/Mask_RCNN) is built with TensorFlow 1.15 which is now no longer supported and it has been updated by TensorFlow 2.

### Training the model

We trained a Mask RCNN model with the JuniperMapper dataset, using the Photo Interpreted images. The annotations from JuniperMapper must be converted into a valid Mask RCNN format. Use the following code:

```bash
python convert_annotations.py
```

Then, use the `Notebooks/shrubs_train_experiments.ipynb` to trainthe model.

TODO: Explain the training experiments to detect the best Matterport's Masked R-CNN configuration to detect shrubs.

Model was trained using a machine with:

| Component name | Model |
| -------------- | ----- |
| Processor | Intel(R) Core(TM) i7-14700KF (3.40 GHz) |
| RAM instalada | 64 GB |
| Graphic card | NVIDIA GeForce RTX 5070 Ti (16 GB) |
| OS | Windows 11 x64 |

### Custom `shrub_matterport` package

*Describe the custom classess to handle the data to classify and validate bin/shrubs/utils.py*


## Detectron2 Mask R-CNN

Replicate the model trained in [Khaldi et al. 2024](https://doi.org/10.1016/j.jag.2024.104191) paper.

### Installation

The most reliable method to install it on Windows is to run Detectron2 through WSL2 with Ubuntu, rather than compiling it as a native Windows application.

Detectron2's documentation still lists Linux and macOS as supported systems. It notes that Windows compatibility is tested to some extent, but the project does not provide official Windows support or official Windows binaries.

WSL2 provides a Linux environment inside Windows. It also supports NVIDIA CUDA acceleration, so Detectron2 can use a compatible NVIDIA GPU. See [Enable NVIDIA CUDA on WSL](https://learn.microsoft.com/en-us/windows/ai/directml/gpu-cuda-in-wsl).

#### 1. Install WSL2 and Ubuntu.

Open PowerShell as Administrator and run:

```PowerShell
wsl --install
```

The default installation normally includes Ubuntu. Then confirm that WSL2 is being used:

```PowerShell
wsl --list --verbose
```

If not, install Ubuntu 22.04 LTS:

```
wsl --install Ubuntu-22.04
```

> Why v22.04? Mature ecosystem and fewer problems with older ML dependencies.

Restart Windows when requested.

If Ubuntu shows version 1:

```PowerShell
wsl --set-version Ubuntu 2
```

#### 2. Prepare Ubuntu.

Open Ubuntu from the Start menu:

```bash
sudo apt update
sudo apt upgrade -y

sudo apt install -y \
    build-essential \
    git \
    python3 \
    python3-pip \
    python3-venv \
    ninja-build \
    libgl1
```

Then, create an isolated Python environment:

```bash
python3 -m venv ~/venvs/detectron2
source ~/venvs/detectron2/bin/activate

python -m pip install --upgrade pip setuptools wheel
```

#### 3. Install PyTorch

Get the installation command from the [PyTorch webpage](https://pytorch.org/get-started/locally/). Select the following options:

- Stable
- Linux
- Pip
- Python
- CPU, or the CUDA version appropriate for your system. For NVIDIA graphics cards, check the [NVIDIA CUDA compatibility webpage](https://developer.nvidia.com/cuda/gpus).
    - We've used CUDA 13.0 because we are using an NVIDIA GeForce RTX 5070 Ti (16 GB).


In our system:

```bash
pip3 install torch torchvision
```

After installation, verify it:

```bash
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

For GPU use, install the current NVIDIA Windows driver that supports CUDA in WSL. Do **not** install a separate Linux NVIDIA display driver inside Ubuntu; WSL accesses the Windows driver. See [Enable NVIDIA CUDA on WSL 2](https://learn.microsoft.com/en-us/windows/ai/directml/gpu-cuda-in-wsl) from Microsoft Learn platform, and [Getting Started with CUDA on WSL 2](https://docs.nvidia.com/cuda/wsl-user-guide/index.html#getting-started-with-cuda-on-wsl).

#### 4. Install Detectron2

Build it from the current repository:

```bash
cd /mnt/c/projects
git clone https://github.com/facebookresearch/detectron2.git
cd detectron2
python -m pip install --no-build-isolation -e .
```

Detectron2 must be compiled against the installed PyTorch version. Reinstall or rebuild Detectron2 after making major changes to PyTorch or CUDA.

#### 5. Verify the installation

```bash
python -c "import detectron2; print('Detectron2 imported successfully')"
python -m detectron2.utils.collect_env
```

Note that, despite this environment report:

```text
CUDA compiler: not available
CUDA_HOME: None - invalid!
```

Your current Detectron2 installation already contains functional CUDA extensions. You can proceed with basic GPU training. The missing `CUDA_HOME` means you currently cannot easily recompile Detectron2 or other CUDA extensions. It does not prevent you from using the CUDA extension that has already been compiled successfully.

#### 6. Fix possible errors

Detectron2 source imports `pkg_resources`, but recent setuptools versions, starting with version 82, no longer include that module. If an error like this appears:

```text
ModuleNotFoundError: No module named 'pkg_resources'
```

Run the following line:

```bash
python -m pip install "setuptools<82"
```

### Train the model

In order to reproduce the original paper and get the weights of the model, first the JuniperMapper labels have to be converted. They should have worked as it is, but they don't follo COCO semantics (as Detectron2 requires). Run the `detectron2_convert_annotations.py` file in order to produce valid instances, i.e., one annotation dictionary equals one instance.

```python
python detectron2_convert_annotations.py ^
  --images-dir D:/dataset/Test/Images ^
  --shapefiles-dir D:/dataset/Test/Shapefiles ^
  --output-json D:/dataset/Test/annotations/Test.json
```

Then, the `detectron_convert_annotations.py` code have been used to train the model following the parameters used by Khaldi et al., using the training set to optimize model weights and the validation set to select checkpoint and optionally tune training configuration.

```python
python detectron2_shrubs_train.py
```

All the variables are already set up inside the script. If you want to modify them, open the python file and change the global variables containing the paths of the training data and the folder to store the weights.

### Extract the validation metrics

After training the model, the statistics are extracted using the PI test data and the FW instances. It mimics the paper's metrics and graphics to allow a comparisson.

```python
python evaluate_detectron2.py ^
  --weights model_best.pth ^
  --dataset shrubs_test_pi ^
    ./JuniperMapper/Photo_Interpretation_Data/Test/Annotations/Test.json ^
    ./JuniperMapper/Photo_Interpretation_Data/Test/Images ^
  --dataset shrubs_test_fw ^
    ./JuniperMapper/Field_Work_Data/External_Val_Data/Annotations/Val.json ^
    ./JuniperMapper/Field_Work_Data/External_Val_Data/Images ^
  --output-dir ./project/results/evaluation/model_detectron2
```


## Funding

