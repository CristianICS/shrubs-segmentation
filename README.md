## Introduction

Instance segmentation-based approach to shrub delineation in burnt areas. The aim of this repository is perform an study on how transfer learning and Deep Learning models can classify shrubs in a fire affected area.

Models used:

- Matterport Mask R-CNN with hyperparameter tunning via three steps experiment;
- Detectron2 Mask R-CNN architecture with hyperparamenters proposed by [Khaldi et al. 2024](https://doi.org/10.1016/j.jag.2024.104191)
- SAM3 model using zero-learning and few-shots classification

## Data

The models have trained with JuniperMapper dataset created by [Khaldi et al. (2023)](https://zenodo.org/records/10058715). The aim of the study is to test if the dataset could be valid to segmentate other type of shrubs in a different environment.

The area of study is a hillside burnt in 2003, near Encinacorba (Spain). The data was collected with a Zenmuse L1 camera, obtaining both RGB images with a spatial resolution of 5 centimeters and a LiDAR point cloud. A field campaign was carried to collect a small shrub dataset to validate the models.

## Reproduce the research

Download the current repository.

```bash
git clone https://github.com/CristianICS/shrubs-segmentation
```

With conda installed from any of the official sources ([Dowload Miniconda](https://docs.conda.io/en/latest/miniconda.html)), go to the project directory:

```bash
cd /d/shrubs-segmentation
```

Install and set up the environment to run the code related with `geopandas` and `rasterio` packages. This isolated environment avoids conflict between the dependencies used to train the models. Run the following line:

> Note: Using [mamba](https://github.com/conda-forge/miniforge) will shorten the installation times.

```bash
conda env create -f environment.yml -y
```

The process has been done using a machine with:

| Component name | Model |
| -------------- | ----- |
| Processor | Intel(R) Core(TM) i7-14700KF (3.40 GHz) |
| RAM instalada | 64 GB |
| Graphic card | NVIDIA GeForce RTX 5070 Ti (16 GB) |
| OS | Windows 11 x64 |

The following environments must be created following the next instructions:

- `mrcnn_tf2` (Matterport, native Windows, no rasterio)
- `~/venvs/detectron2` (Detectron2, WSL2 Ubuntu-22.04),
- `sam3_shrubs` (SAM 3, WSL2 Ubuntu-22.04)
- `mrcnn_masks` geopandas/rasterio env pre-tiling for Matterport, plus merging and evaluation.

## Matterport Mask R-CNN

The `bin` folder contains the `mrcnn` module from [Ahmed Gad's Mask-RCNN-TF2 repository](https://github.com/ahmedfgad/Mask-RCNN-TF2). It is used to apply the model Mask R-CNN to the dataset.

> The original [Matterport Mask RCNN repo](https://github.com/matterport/Mask_RCNN) is built with TensorFlow 1.15 which is now no longer supported and it has been updated by TensorFlow 2.

Install and set up the environment with the correct dependencies. Run the following line:

```bash
conda create -n mrcnn_tf2 -c conda-forge -y python=3.7
```

> Note: Python 3.7 is an stable version used to test the implementation.

Then, activate the environment and install the dependencies using `pip`:

```bash
conda activate mrcnn_tf2
python -m pip install -r requirements.txt
```

If the environment were installed correctly, the jupyter notebooks containing the operations to apply and validate the Matterport Mask R-CNN can be executed.

### Training the model

We trained a Mask RCNN model with the JuniperMapper dataset, using the Photo Interpreted images. The annotations from JuniperMapper must be converted into a valid Mask RCNN format. Use the following code:

```bash
python matterport_convert_annotations.py
```

The code creates a custom `json` file with the valid format inside the `JuniperMapper` folder, without removing or updating any of the original files. Then, use `Notebooks/shrubs_train_experiments.ipynb` notebook to train the model. It runs a sequential, three-phase testing design instead of a full hyperparameter grid, which would be costly and more likely to pick a configuration that only looks good by chance.

Phase A compares training methods (heads-only vs. staged fine-tuning, with and without geometric augmentation), Phase B tunes the learning rate on top of the winning Phase A setup, and Phase C compares the ResNet-50 backbone against the ResNet-101 baseline using the winning Phase A+B setup. After each phase, every candidate is evaluated on the validation split and the one with the highest mask mAP (IoU 0.50–0.95) is preserved. The final winning configuration, its training schedule, augmentation choice, and validation metrics (mask mAP, AP50, AP75, confidence threshold, F1) are saved to `data/logs/matterport/best_model_configuration.json`.

### Validation metrics

In order to produce the graphics to validate the training, use the `Notebooks/shrubs_evaluation.ipynb` notebook. It produces the graphics with the same metrics used by [Khaldi et al. (2024)](https://doi.org/10.1016/j.jag.2024.104191).

### Run inference on unlabeled images

To segment a plain set of RGB images with a trained checkpoint, it is mandatory to tile the original RGB mosaic first.Matterport's `mrcnn_tf2` conda env has a Windows DLL conflict between its old TensorFlow/Keras stack and GDAL (rasterio fails to import there even after a clean reinstall). Use the following code inside `mrcnn_masks` environment:

```bash
python Python/tile_orthomosaic.py \
  --orthomosaic data/rasters/orthomosaic_clip.tif \
  --output data/rasters/deployment_tiles/matterport \
  --tile-size 448 --overlap 64
```

This produces plain `<stem>_r<row>_c<col>.png` + `.json` transform sidecar pairs.

After tilling operation, run the following code to run the inference using the current best checkpoint (`data/logs/matterport/best_model_configuration.json`). Run the code inside `mrcnn_tf2` conda environment:

```bash
python matterport_deploy.py \
  --tiles data/rasters/deployment_tiles/matterport \
  --output results/deployment/tile_masks/matterport \
  --confidence 0.50
```

Then, to produce the final shrub masks, use `refine_detections.py` inside the `masks_mrcnn` environment:

```bash
python refine_detections.py \
  --predictions results/deployment/tile_masks/matterport \
  --output results/deployment/shrub_map_matterport.geojson \
  --crs EPSG:25830
```

## Detectron2 Mask R-CNN

The Detectron2 implementation of the Mask R-CNN is the one used in [Khaldi et al. 2024](https://doi.org/10.1016/j.jag.2024.104191) paper.

The most reliable method to install it on Windows is to run Detectron2 through WSL2 with Ubuntu, rather than compiling it as a native Windows application. Detectron2's documentation lists Linux and macOS as supported systems. It notes that Windows compatibility is tested to some extent, but the project does not provide official Windows support or official Windows binaries.

WSL2 provides a Linux environment inside Windows. It also supports NVIDIA CUDA acceleration, so Detectron2 can use a compatible NVIDIA GPU. See [Enable NVIDIA CUDA on WSL](https://learn.microsoft.com/en-us/windows/ai/directml/gpu-cuda-in-wsl).

### Windows installation

The following details contain the instructions to install Detectron2 on WSL.

<details>
<summary>Instructions</summary>

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

**Configure Ubuntu**

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

**Install PyTorch**

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

**Install Detectron2**

Build it from the current repository:

```bash
cd /mnt/c/desired_folder
git clone https://github.com/facebookresearch/detectron2.git
cd detectron2
python -m pip install --no-build-isolation -e .
```

Detectron2 must be compiled against the installed PyTorch version. Reinstall or rebuild Detectron2 after making major changes to PyTorch or CUDA.

**Verify the installation**

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

**Fix possible errors**

Detectron2 source imports `pkg_resources`, but recent setuptools versions, starting with version 82, no longer include that module. If an error like this appears:

```text
ModuleNotFoundError: No module named 'pkg_resources'
```

Run the following line:

```bash
python -m pip install "setuptools<82"
```

</details>

### Training phase

In order to reproduce the original paper and get the weights of the model, first the JuniperMapper labels have to be converted. Run the `detectron2_convert_annotations.py` inside the `mrcnn_masks` conda environment in order to produce valid instances, i.e., one annotation dictionary equals one instance.

```python
python detectron2_convert_annotations.py ^
  --images-dir ./data/Test/Images ^
  --shapefiles-dir ./data/Test/Shapefiles ^
  --output-json ./data/Test/annotations/Test.json
```

Then, train the model following the parameters used by Khaldi et al. 2024, using the training set to optimize model weights and the validation set to select checkpoint and optionally tune training configuration. Inside the WSL environment run:

```python
python detectron2_shrubs_train.py
```

All the variables are already set up inside the script. If you want to modify them, open the python file and change the global variables containing the paths of the training data and the folder to store the weights.

### Validation metrics

After training the model, the statistics are extracted using the PI test data and the FW instances. It mimics the paper's metrics and graphics to allow a comparisson.

```bash
python detectron2_evaluate.py \
  --weights model_best.pth \
  --dataset shrubs_test_pi \
    data/JuniperMapper/Photo_Interpretation_Data/Test/Annotations/Test.json \
    data/JuniperMapper/Photo_Interpretation_Data/Test/Images \
  --dataset shrubs_test_fw \
    data/JuniperMapper/Field_Work_Data/External_Val_Data/Annotations/Val.json \
    data/JuniperMapper/Field_Work_Data/External_Val_Data/Images \
  --output-dir ./results/evaluation/model_detectron2
```

### Run inference on unlabeled images

To segment a georeferenced mosaic and produce the shrub masks, use the following code inside the `detectron2` virtual environment:

```bash
python detectron2_deploy.py \
  --orthomosaic data/rasters/orthomosaic_clip.tif \
  --weights data/logs/detectron2/model_best.pth \
  --output results/deployment/tile_masks/detectron2 \
  --confidence 0.35
```

> Note that pretilling operation is not required here because the model internally perform the tiling.

Then, to produce the final shrub masks, use `refine_detections.py` inside the `masks_mrcnn` environment:

```bash
python refine_detections.py \
  --predictions results/deployment/tile_masks/detectron2 \
  --output results/deployment/shrub_map_detectron2.geojson \
  --crs EPSG:25830
```

Note: to segment a plain set of RGB images with a trained checkpoint, with no COCO ground-truth annotations nor georeferenced image required, use `detectron2_predict.py`:

```bash
python Python/detectron2_predict.py \
  --weights data/logs/detectron2/model_best.pth \
  --input data/unlabeled_images \
  --output results/inference/model_detectron2
```

For each image it writes a binary shrub mask (`*_shrubs.png`), a per-pixel confidence map (`*_shrub_confidence.png`), and a CSV of instance scores and boxes (`*_shrub_instances.csv`). Adjust `--confidence` to change the score threshold used to keep a prediction (default `0.5`).

## SAM 3

The Segment Anything Model 3 (SAM 3) developed by Meta is used to test its zero-shot accuracy on shrub segmentation, i.e., running the pretrained foundation checkpoint with a text prompt and no shrub-specific fine-tuning; but also its few-shot classification schema using the JuniperMapper dataset to fine tunning the model.

### Installation

The recommended installation **uses conda inside WSL** to create a new environment with the following line:

```bash
cd /mnt/c/shrubs-segmentation
conda env create -f environment_sam3.yml
```

<details>
<summary>Check this instructions to see how to install conda inside WSL</summary>

For WSL, Miniconda is a good choice because it provides Conda without installing a large collection of packages. [Conda's current documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html?highlight=miniconda) recommends Miniconda or Miniforge for this purpose.

**Open the WSL terminal**

First, check the CPU architecture:

```bash
uname -m
```

If it says:

```text
x86_64
```

continue with the commands below. If it says `aarch64` or `arm64`, you must need to use the ARM installer command.

**Download Miniconda**

Run:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

If `wget` isn't installed:

```bash
sudo apt update
sudo apt install wget -y
```

Then run the download command again.

**Run the installer**

```bash
bash Miniconda3-latest-Linux-x86_64.sh
```

The [official Conda instructions](https://docs.conda.io/_/downloads/conda/en/stable/pdf) use this Linux installer approach. ([docs.conda.io][2])

Read the license agreement and press `Enter` to scroll through it. When the text *Do you accept the license terms?* appears, type `yes` and press `Enter`.

**Accept the default installation location**

Press `Enter` when the following text appears:

```text
Miniconda3 will now be installed into this location:

/home/yourname/miniconda3
```

> Don't use `sudo` for the Miniconda installation itself. Installing it under your home directory means administrator/root permissions aren't required.

**Avoid letting Conda initialize your shell**

Near the end, the installer will ask something like:

```text
Do you wish to update your shell profile to automatically initialize conda?
```

Type:

```text
no
```

Then press Enter. If you accept this feature, every time you open WSL console conda base environment will be initialized.

After this step, reload WSL console by closing the terminal and open a new one.

If you run `conda --version` and the actual conda version appears, conda has been installed correctly.
</details><br>

Inside the new environment, authenticate with Hugging Face before the first run so the gated checkpoint can be downloaded:

```bash
conda activate sam3_shrubs
hf auth login
```

### Training the model

The code `Python/sam3_domain_adapt.py` compares the zero-shots approachs with the model fine tunning using the shrubs inside JuniperMapper dataset. Zero-shot tends to over-segmentate, so this approach aims to fine-tunes only SAM 3's transformer/detection/segmentation head on JuniperMapper, with both the vision and language backbones frozen (`shrubs_sam3.training.FrozenBackboneTrainer`), so the trainable parameter count stays small relative to the dataset:

```bash
python Python/sam3_domain_adapt.py \
  --output data/logs/sam3/domain_adapt --max-epochs 30
```

By default it trains on the JuniperMapper Photo Interpretation Train split and validates on its Val split (override with `--train-json`/`--train-images`/`--val-json`/`--val-images`); `--batch-size`, `--num-workers`, `--checkpoint-freq`, and `--device` cover the rest of the common tuning knobs (`python Python/sam3_domain_adapt.py --help` for the full list). Pass `--checkpoint` (the same local release checkpoint file `sam3_deploy.py`/`sam3_evaluate.py` take) to start training from an already-downloaded checkpoint instead of re-downloading the gated one from Hugging Face. It writes `checkpoint.pt` (latest epoch), periodic `checkpoint_<N>.pt` snapshots, and a `val_shrubs_val_detection_<metric>.pt` snapshot whenever that bbox metric (e.g. AP, AP50) improves, all under `<output>/checkpoints`. None of these is chosen automatically as the model to deploy. That in-loop meter only tracks box AP, not mask quality. So the produced checkpoints were compared using `sam3_evaluate.py`'s instance-IoU/S-IoU metrics:

```bash
python Python/sam3_evaluate.py \
  --checkpoint data/logs/sam3/domain_adapt/checkpoints/checkpoint.pt \
  --dataset shrubs_test_fw \
    data/JuniperMapper/Field_Work_Data/External_Val_Data/Annotations/Val.json \
    data/JuniperMapper/Field_Work_Data/External_Val_Data/Images \
  --output-dir results/evaluation/model_sam3/domain_adapted
```

Repeat against each candidate checkpoint.

### Run inference

Use the `Python/sam3_deploy.py` CLI to predict the shrubs in the study area by pointing `--checkpoint` at the one yielding the best F1 score (`build_processor` auto-detects a fine-tuned checkpoint vs. an official release one). The script tiles a GeoTIFF (or every GeoTIFF in a folder), runs SAM 3 with a text prompt on each tile, and stitches the results back into georeferenced outputs:

```bash
python Python/sam3_deploy.py data/rasters/orthomosaic_clip.tif \
  --output results/deployment/tile_masks/sam3/shrub --prompt shrub \
  --checkpoint data/logs/sam3/domain_adapt/checkpoints/<checkpoint>.pt
```

The model has been run with the following prompts to compare semantically similar prompts while keeping all other settings fixed:

- `brown_shrub`
- `burned_shrub`
- `bush`
- `dry_shrub`
- `low_shrub`
- `mediterranean_shrub`
- `shrub`
- `small_bush`
- `small_shrub`
- `woody_shrub`

> Note that the output of each prompt is stored inside a folder with their text.

Run `python Python/sam3_deploy.py --help` for the remaining options (`--confidence`, `--rgb-bands`, `--tile-size`, `--overlap`, `--overwrite`).

The code writes a binary shrub GeoTIFF, a maximum-confidence GeoTIFF, and a CSV of tile-level instances.

After inference, the masks must be refined as in the previous models using the `Python/sam3_refine_detections.py` script. Run it inside the `mrcnn_masks` environment since it needs `geopandas`:

```bash
python Python/sam3_refine_detections.py \
  results/deployment/tile_masks/sam3/<prompt> \
  --output results/deployment/<prompt>_map_sam3.geojson \
  --crs EPSG:25830
```

## Accuracy assessment

The three trained models (Detectron2 Mask R-CNN, Matterport Mask R-CNN, SAM 3) have been applied over the RGB orthomosaic using the following models' operating points:

| Model | Weights / checkpoint | Confidence threshold |
| --- | --- | --- |
| Detectron2 Mask R-CNN | `data/logs/detectron2/model_best.pth` | 0.35 |
| Matterport Mask R-CNN | The winning config in `best_model_configuration.json` | 0.50 |
| SAM 3 (domain-adapted) | `data/logs/sam3/domain_adapt/checkpoints/checkpoint_10.pt` | 0.60 |

After the deployment phase, each model's shrub map is scored against the manually digitized over the field work plots (`data/encinacorba_annotations.gpkg - square_rois` layer). The `field_work_masks` truth layer contains 79 polygons: 58 shrub masks (`shrub == 1`, mean 3.2 square meters) and 21 tree masks to assess false positives (`shrub == 0`, mean 15.2 square meters). It also contains an `esp_code` species code. It only covers a ~261x278 m plot inside the orthomosaic. It's **exhaustive shrub digitization** within that plot (every shrub inside it was digitized), allowing the computation of the precision/recall metrics.

See [the `results/deployment/README.md`](./results/deployment/README.md) for more information.

## LiDAR refinement

## Licenses

This repository combines material under different licenses. Each component keeps its own terms:

| Component | License | Notes |
| --------- | ------- | ----- |
| Code in this repository (`Python/`, `R/`, `bin/shrubs_*`, notebooks) | [MIT](LICENSE) | |
| `bin/mrcnn/` | [MIT, Matterport](bin/mrcnn/LICENSE) | Taken from [Ahmed Gad's Mask-RCNN-TF2](https://github.com/ahmedfgad/Mask-RCNN-TF2), a TensorFlow 2 port of [Matterport's Mask R-CNN](https://github.com/matterport/Mask_RCNN). Modified; see [`bin/mrcnn/README.md`](bin/mrcnn/README.md) for the list of changes. |
| `bin/shrubs_sam3/configs/juniper_frozen_backbone.yaml` | [SAM License](bin/shrubs_sam3/configs/LICENSE) (Meta) | Adapted from the fine-tuning config of the SAM 3 repository, so it is a derivative of SAM Materials and stays under the SAM License, not the MIT license above. |
| Field data (`data/encinacorba_annotations.gpkg`, `data/spectral_signatures/`) | CC BY 4.0 | Attribute this repository (see [`CITATION.cff`](CITATION.cff)). |

Third-party software and data that are **not** included in this repository must be obtained under their own licenses:

- **[Detectron2](https://github.com/facebookresearch/detectron2)** (Apache 2.0): installed from its repository, as described above.
- **[SAM 3](https://github.com/facebookresearch/sam3)**: installed with `environment_sam3.yml`. Its checkpoint is gated on Hugging Face and requires accepting the [SAM License](https://github.com/facebookresearch/sam3/blob/main/LICENSE). Fine-tuned SAM 3 checkpoints are derivatives of SAM Materials, so they are not distributed with this repository, and any that are shared later must be shared under that license. Publications using SAM 3 must acknowledge its use.
- **[JuniperMapper](https://zenodo.org/records/10058715)** (CC BY 4.0, Khaldi et al. 2023): download it from Zenodo into `data/JuniperMapper/`. It is not redistributed here. Cite the dataset and [Khaldi et al. 2024](https://doi.org/10.1016/j.jag.2024.104191) when using it or models trained on it.

## Citation

If you use this code, please cite it with the metadata in [`CITATION.cff`](CITATION.cff) (GitHub shows a "Cite this repository" button from it), together with the JuniperMapper dataset and paper.

## Funding

