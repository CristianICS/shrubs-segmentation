## Introduction

The current repo contains the code and workflow to reproduce the paper:

> Transferability of Deep Learning Models for Post-Fire Shrub Mapping Using UAV RGB and LiDAR Data

The study implements an instance segmentation-based approach to shrub delineation in burnt areas. The aim is perform a series of trainings over a small and controlled and previously burned area to assess how transfer learning and Deep Learning models can classify shrubs in a fire affected area.

The following models are used:

- Matterport Mask R-CNN with hyperparameter tunning;
- Detectron2 Mask R-CNN architecture with hyperparamenters proposed by [Khaldi et al. 2024](https://doi.org/10.1016/j.jag.2024.104191);
- SAM 3 model, both zero-shot and domain-adapted.

## Data

The training is conducted with JuniperMapper dataset, created by [Khaldi et al. (2023)](https://zenodo.org/records/10058715).

After training phase, the model is deployed creating a shrub mask over the study area, a hillside burnt in 2003, near Encinacorba (Spain). The orthomosaic obtained with the RGB images collected with a Zenmuse L1 camera is used. The spatial resolution is 5 cm.

The segmented masks were assessed with a manually digitized masks, inspected during a field campaign. The LiDAR point cloud collected from the drone is used to refine the masks (see [refinement section](#lidar-refinement))

One of the aims of the study is to test if a shrub dataset outside the study area could be valid to segmentate other type of shrubs in a different environment.

### Data availability

The field annotations (`data/encinacorba_annotations.gpkg`) are included in this repository. The `field_work_masks` layer holds the digitized reference masks, with each mask's species and growth form. The `square_rois` layer holds the 26 field plots covered by the orthomosaic, with the GNSS position and orthometric height of each plot centre.

The following files are not included because of their size. They are available from the corresponding author on reasonable request. Place them at the paths below so the scripts find them with their default arguments.

| File | Path | Used by |
| ---- | ---- | ------- |
| RGB orthomosaic (5 cm) | `data/rasters/orthomosaic_clip.tif` | Deployment, DTM grid |
| LiDAR point cloud | `data/point_cloud/encinacorba_cloud.laz` | Ground classification, DTM and CHM ([R/README.md](R/README.md)) |
| Canopy height model (0.25 m) | `data/rasters/encinacorba_chm.tif` | LiDAR refinement. Can be rebuilt from the point cloud |
| Detectron2 weights | `data/logs/detectron2/model_best.pth` | Detectron2 validation and deployment |
| Matterport weights | `data/logs/matterport/phase3_resnet50_seed20260803_20260816T204953/best_validation_map_checkpoint.h5` | Matterport validation and deployment (path listed in `data/logs/matterport/best_model_configuration.json`) |
| SAM 3 domain-adapted checkpoint | `data/logs/sam3/domain_adapt/checkpoints/model_best.pt` | SAM 3 validation and deployment. Shared only under the SAM License (see [Licenses](#licenses)) |

The JuniperMapper training dataset must be downloaded from [Zenodo](https://zenodo.org/records/10058715) (see [Licenses](#licenses)).

## First steps

Download the current repository.

```bash
git clone https://github.com/CristianICS/shrubs-segmentation
```

With conda installed from any of the official sources ([Dowload Miniconda](https://docs.conda.io/en/latest/miniconda.html)), go to the project directory:

```bash
cd /d/shrubs-segmentation
```

Install and set up the `mrcnn_masks` environment to run the code related with `geopandas` and `rasterio` packages. This isolated environment avoids conflict between the dependencies used to train the models. Run the following line:

> Note: Using [mamba](https://github.com/conda-forge/miniforge) will shorten the installation times.

```bash
conda env create -f environment.yml -y
```

The process has been done using a machine with:

| Component name | Model |
| -------------- | ----- |
| Processor | Intel(R) Core(TM) i7-14700KF (3.40 GHz) |
| RAM | 64 GB |
| Graphic card | NVIDIA GeForce RTX 5070 Ti (16 GB) |
| OS | Windows 11 x64 |

The next steps must be followed to configure the required environments. After completing all the steps, the following environments must be created:

- `mrcnn_tf2` (Matterport, native Windows, no rasterio)
- `~/venvs/detectron2` (Detectron2, WSL2 Ubuntu-22.04),
- `sam3_shrubs` (SAM 3, WSL2 Ubuntu-22.04)
- `mrcnn_masks` geopandas/rasterio env pre-tiling for Matterport, plus merging and evaluation. -->

### Matterport Mask R-CNN

The `bin` folder of this repository contains the `mrcnn` module from [Ahmed Gad's Mask-RCNN-TF2](https://github.com/ahmedfgad/Mask-RCNN-TF2). It is used to implement the Mask R-CNN Matterport configuration.

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

### Detectron2 Mask R-CNN

The Detectron2 implementation of the Mask R-CNN is the one used in [Khaldi et al. 2024](https://doi.org/10.1016/j.jag.2024.104191) paper.

The most reliable method to install it on Windows is to run Detectron2 through WSL2 with Ubuntu, rather than compiling it as a native Windows application. Detectron2's documentation lists Linux and macOS as supported systems. WSL2 provides a Linux environment inside Windows. It also supports NVIDIA CUDA acceleration, so Detectron2 can use a compatible NVIDIA GPU. See [Enable NVIDIA CUDA on WSL](https://learn.microsoft.com/en-us/windows/ai/directml/gpu-cuda-in-wsl).

Follow the next instructions to install Detectron2 on WSL.

#### Install WSL

Open PowerShell as administrator and run:

```PowerShell
wsl --install
```

The default installation normally includes Ubuntu. Confirm that WSL2 is being used:

```PowerShell
wsl --list --verbose
```

Otherwise, install Ubuntu 22.04 LTS:

```
wsl --install Ubuntu-22.04
```

> Why v22.04? Mature ecosystem and fewer problems with older ML dependencies.

Restart Windows when requested.

If Ubuntu shows version 1:

```PowerShell
wsl --set-version Ubuntu 2
```

#### Configure Ubuntu

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

#### Create the environment

Create an isolated Python environment with this code:

```bash
python3 -m venv ~/venvs/detectron2
source ~/venvs/detectron2/bin/activate

python -m pip install --upgrade pip setuptools wheel
```

Then, install PyTorch. Get the installation command from the [PyTorch webpage](https://pytorch.org/get-started/locally/). Select the following options:

- Stable
- Linux
- Pip
- Python
- CPU, or the CUDA version appropriate for your system. For NVIDIA graphics cards, check the [NVIDIA CUDA compatibility webpage](https://developer.nvidia.com/cuda/gpus).
    - We've used CUDA 13.0 because we are using an NVIDIA GeForce RTX 5070 Ti (16 GB).

Run the installation command:

```bash
pip3 install torch torchvision
```

After installation, verify it:

```bash
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

For GPU use, install the current NVIDIA Windows driver that supports CUDA in WSL. Do **not** install a separate Linux NVIDIA display driver inside Ubuntu; WSL accesses the Windows driver. See [Enable NVIDIA CUDA on WSL 2](https://learn.microsoft.com/en-us/windows/ai/directml/gpu-cuda-in-wsl) from Microsoft Learn platform, and [Getting Started with CUDA on WSL 2](https://docs.nvidia.com/cuda/wsl-user-guide/index.html#getting-started-with-cuda-on-wsl).

#### Install Detectron2

First, clone the official repo somewhere in your system:

```bash
cd /mnt/c/desired_folder
git clone https://github.com/facebookresearch/detectron2.git
```

Then, enter the official repository and install their dependencies.

> Important: you must be inside the environment created in the previous step.

```bash
cd detectron2
python -m pip install --no-build-isolation -e .
```

After the install, Detectron2 must be compiled against the installed PyTorch version. Reinstall or rebuild Detectron2 after making major changes to PyTorch or CUDA.

Verify the installation with this code:

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

#### Install the project dependencies

Detectron2 does not install `rasterio` or OpenCV, which `detectron2_deploy.py` needs to tile the orthomosaic and extract the mask polygons:

```bash
python -m pip install rasterio opencv-python-headless
```

#### Fix possible errors

Detectron2 source imports `pkg_resources`, but recent setuptools versions, starting with version 82, no longer include that module. If an error like this appears:

```text
ModuleNotFoundError: No module named 'pkg_resources'
```

Run the following line:

```bash
python -m pip install "setuptools<82"
```

### SAM 3

The Segment Anything Model 3 (SAM 3) was developed by Meta. It is used to test its zero-shot accuracy on shrub segmentation, i.e., running the pretrained foundation checkpoint with a text prompt and no shrub-specific fine-tuning. It is then domain-adapted by fine-tuning its decoder heads on the full JuniperMapper training split, with the vision and language backbones kept frozen.

Before to start, install the WSL console following [the instructions inside Detectron2 installation section](#install-wsl).

#### Install conda

The recommended installation is using conda inside WSL to create a new environment.

For WSL, Miniconda is a good choice because it provides Conda without installing a large collection of packages. [Conda's current documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html?highlight=miniconda) recommends Miniconda or Miniforge for this purpose.

Inside the WSL terminal, check the CPU architecture:

```bash
uname -m
```

If it says:

```text
x86_64
```

continue with the installation. If it says `aarch64` or `arm64`, you must need to use the ARM installer command.

Download and install Miniconda running:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

Note: If `wget` isn't installed execute:

```bash
sudo apt update
sudo apt install wget -y
```

Then run the download command again. Then, run the Miniconda installer:

```bash
bash Miniconda3-latest-Linux-x86_64.sh
```

The [official Conda instructions](https://docs.conda.io/_/downloads/conda/en/stable/pdf) use the above Linux installer approach. Read the license agreement and press `Enter` to scroll through it. When the text *Do you accept the license terms?* appears, type `yes` and press `Enter`.

Accept the default installation location pressing `Enter` when the following text appears:

```text
Miniconda3 will now be installed into this location:

/home/yourname/miniconda3
```

> Don't use `sudo` for the Miniconda installation itself. Installing it under your home directory means administrator/root permissions aren't required.

After confirming the directory, avoid letting Conda initialize your shell. Near the end, the installer will ask something like:

```text
Do you wish to update your shell profile to automatically initialize conda?
```

Type `no` and press Enter. If you accept this feature, every time you open WSL console conda base environment will be initialized.

Reload WSL console by closing the terminal and open a new one. If you run `conda --version` and the actual conda version appears, conda has been installed correctly.

#### Download the model weights

Open WSL and go to the repository path:

```bash
cd /mnt/c/shrubs-segmentation
```

The first step is create the conda environment with the following line:

```bash
conda env create -f environment_sam3.yml
```

Then point the environment's library path at its own `lib` folder, and reactivate it so the change takes effect:

```bash
conda activate sam3_shrubs
conda env config vars set LD_LIBRARY_PATH=$CONDA_PREFIX/lib
conda deactivate
```

> Note: The pip-installed PyTorch wheel otherwise loads Ubuntu's system `libstdc++`, which is older than the one conda-forge's matplotlib needs. Scripts that import `torch` before matplotlib then fail with `ImportError: ... version 'CXXABI_1.3.15' not found`. Whether a script hits this depends on import order, so set the variable even if everything works at first.

Inside the new environment, authenticate with Hugging Face before the first run so the gated checkpoint can be downloaded:

```bash
conda activate sam3_shrubs
hf auth login
```

## Training phase

We first performed a model selection step, trained the Matterport Mask R-CNN and SAM3 models with different configurations. The JuniperMapper Photo Interpreted images were used, the ones included in the training and validation sets. The test and field-work sets were saved for the final model evaluation.

Every epoch's checkpoint is evaluated on the Photo Interpretation validation split, and the one with the highest mask AP50 is kept as the run's best checkpoint (ties broken by mask mAP50:95, then the earliest epoch). After each phase, the candidate with the highest mask AP50 is preserved. This is the same criterion used for Detectron2 and SAM 3.

### Matterport sequential model selection

Before the training phase, the annotations from JuniperMapper must be converted into a valid format. First, inside the `mrcnn_masks` created in [first steps section](#first-steps), run the following code:

```bash
python Python/matterport_convert_annotations.py
```

It creates a custom `json` file with the valid format inside the `JuniperMapper` folder, without removing or updating any of the original files.

Then, use `Python/matterport_train.py` to perform the model evaluation experiment inside the environment created in the [Matterport Mask R-CNN first steps section](#matterport-mask-r-cnn). It runs a sequential, three-phase testing design instead of a full hyperparameter grid, which would be costly and more likely to pick a configuration that only looks good by chance.

- Phase A: compares training methods (heads-only vs. staged fine-tuning, with and without geometric augmentation);
- Phase B: tunes the learning rate on top of the winning Phase A setup;
- Phase C: compares the ResNet-50 backbone against the ResNet-101 baseline using the winning Phase A+B setup.

The final winning configuration, its training schedule, augmentation choice, and validation metrics (mask mAP, AP50, AP75, confidence threshold, F1) are saved to `data/logs/matterport/best_model_configuration.json`.

> Note: a run that stopped partway through isn't counted as completed, so the script starts that experiment again in a new folder.

### Detectron2

Steps to reproduce the original paper and get the weights of the model. Train the model following the parameters used by [Khaldi et al. (2024)](https://doi.org/10.1016/j.jag.2024.104191). Every two epochs, the masks are evaluated on the Photo Interpretation validation split, and the checkpoint with the highest mask AP50 is saved as `model_best.pth` inside `data/logs/detectron2`, the same criterion used for Matterport and SAM 3. Inside the WSL environment run:

```bash
python Python/detectron2_shrubs_train.py
```

All the variables are already set up inside the script. If you want to modify them, open the python file and change the global variables containing the paths of the training data and the folder to store the weights.

### SAM 3 domain adaptation

The code `Python/sam3_domain_adapt.py` performs the domain adaptation using all the shrubs in the JuniperMapper training split and the prompt text `shrub`. Zero-shot SAM 3 tends to over-segment, so this approach fine-tunes only SAM 3's transformer, detection and segmentation head on JuniperMapper, so the trainable parameter count stays small relative to the dataset.

Run the next Python code, and include the path to the original SAM 3 weights with `--checkpoint`. This file is downloaded from HuggingFace using the steps followed in [SAM 3 installation section](#download-the-model-weights).

> Note: Remember to activate the [conda environment `sam3_shrubs`](#download-the-model-weights) before running the script.

```bash
python Python/sam3_domain_adapt.py --checkpoint /path/to/sam3.pt
```

The checkpoint with the highest mask AP50 is saved as `model_best.pt` inside the following path

```text
./data/logs/sam3/domain_adapt/checkpoints
```

Its scores and epoch are written to `model_best.json`.

#### Prompt study

SAM 3 is prompted with text, so the [domain-adapted model](#sam-3-domain-adaptation) is compared across semantically similar prompts on the same data and metric used to domain adapt the model:

- `brown_shrub`
- `burned_shrub`
- `bush`
- `dry_shrub`
- `low_shrub`
- `mediterranean_shrub`
- `small_bush`
- `small_shrub`
- `woody_shrub`

The code `Python/sam3_prompt_study.py` reruns the training-time validation of `model_best.pt`[^1] once per prompt. As during training, the prompt is the category name written into the converted annotations. This makes the `shrub` row reproduce the scores in `model_best.json`:

```bash
python Python/sam3_prompt_study.py
```

[^1]: Same Hydra config, transforms, AMP, and COCO evaluator, through the SAM 3 trainer's `mode=val`.

It defaults to `data/logs/sam3/domain_adapt/checkpoints/model_best.pt` and the ten prompts below. It writes the per-prompt dumps and logs under `data/logs/sam3/prompt_study/<prompt>` and the summary to `results/evaluation/model_sam3/prompt_study_pi_val.csv`. Prompts already scored in the work folder are skipped unless `--overwrite` is passed.

## Validation phase

This step will produce the model internal evaluation using the JuniperMapper Photo Interpretated test set and the masks digitized by experts with ground information inside the Field Work set. The outputs are the graphics with the same metrics used by [Khaldi et al. (2024)](https://doi.org/10.1016/j.jag.2024.104191).

The export files are saved inside the following path:

```text
`results/evaluation/model_<name>/<split>/`
```

The validation phase is used to determine the operational confidence threshold for model deployment. For each model, the threshold is selected from its F1-versus-confidence curve on the Field Work test split, which most closely represents real-world conditions. True positives are defined using S-IoU@50%. The selected operating point is the highest confidence threshold whose F1 score is within 1 percentage point of the maximum F1, thereby favouring more confident predictions and reducing false-positive masks while maintaining near-optimal performance.

The F1 curves used for this selection are the `siou_threshold_50.csv` files in `results/evaluation/model_<model>/shrubs_test_fw/`.

### Matterport

Run the following lines inside the console with the [required `mrcnn_tf2` environment](#matterport-mask-r-cnn):

```bash
conda activate mrcnn_masks
cd c/shrubs-segmentation
python Python/matterport_evaluate.py
```

It rebuilds the model from

```text
`data/logs/matterport/best_model_configuration.json`
``` 

### Detectron2

Create the plots and data to assess Detectron2 configuration by running the following code inside the [Detectron2 environment](#create-the-environment) from the WSL console:

```bash
python Python/detectron2_evaluate.py
```

It selects the weights from

```text
`data/logs/detectron2/model_best.pth`
``` 

### SAM 3

Run the next code inside the [SAM 3 conda environment](#download-the-model-weights) from the WSL console:

```bash
python Python/sam3_evaluate.py
```

It rebuilds the model from

```text
`data/logs/sam3/domain_adapt/model_best.json`
``` 

## Deployment

Run inference on the burnt study area's RGB images.

Only masks with a probability higher than a selected confidence threshold is selected. It depends on each model and it is selected based on the [validation phase](#validation-phase).

| Model | Confidence threshold ($\theta$) |
| ----  | -------------- |
| Matterport Mask R-CNN | 0.60 |
| Detectron2 Mask R-CNN | 0.45 |
| SAM 3 | 0.60 |

Each window produces one `<ortho_stem>_r<row>_c<col>_masks.json` file. It holds the polygon of every predicted instance (masks under 4 px are dropped) together with the window's georeferencing, so no GeoTIFF tile is needed. `Python/refine_detections.py` (inside the [`mrcnn_masks` environment](#first-steps)) merges them into one shrub map per model. Polygons from different windows that overlap, touch or lie within 0.05 m are unioned into one shrub, and the maximum, mean and median of their scores are kept.

> Note: SAM 3 resizes every input to 1008px at inference. Using 448px windows at deployment reproduces that validation scale. Windows are read as raw 8-bit RGB with no contrast stretch, as for the other two models, so the radiometry matches the JuniperMapper training images. The orthomosaic must therefore be 8-bit; the alpha band (`--alpha-band`, default 4) marks empty pixels.

After the first deployment operation, each tile produces one `<ortho_stem>_r<row>_c<col>_masks.json` file with the per-instance mask polygons and the window's georeferencing. In order to merge them into a single shrub map, the `refine_detections.py` code must be applied inside the [`mrcnn_masks` environment](#first-steps).

### Matterport

For Matterport configuration, it is mandatory to tile the original RGB mosaic first. Matterport's `mrcnn_tf2` conda env has a Windows DLL conflict between its old TensorFlow/Keras stack and GDAL, making rasterio fails to import there even after a clean reinstall.

Use the following code inside `mrcnn_masks` environment to tile the original RGB image:

> Note: empty `data/rasters/deployment_tiles/matterport` and `results/deployment/tile_masks/matterport` before rerunning. Files from an older run (another tile size or raster name) aren't overwritten: old tiles would be predicted again and old windows merged into the new map.


```bash
python Python/tile_orthomosaic.py
```

The default behaviour points to `data/rasters/orthomosaic_clip.tif` (RGB mosaic) and the tiles are saved inside `data/rasters/deployment_tiles/matterport`. Those parameters could be overriden with `--orthomosaic` and `--output` arguments.

After tilling operation, run the following code to execute the inference using the current best checkpoint (`data/logs/matterport/best_model_configuration.json`). Run the code inside `mrcnn_tf2` conda environment:

```bash
python Python/matterport_deploy.py
```

Finally, join all the output predictions inside `results/deployment/tile_masks/matterport` with:

```bash
python Python/refine_detections.py \
  --predictions results/deployment/tile_masks/matterport \
  --output results/deployment/shrub_map_matterport.geojson \
  --crs EPSG:25830
```

### Detectron 2

Use the following code inside the `detectron2` virtual environment to segment a georeferenced mosaic and produce the shrub masks:

> Note: empty `results/deployment/tile_masks/detectron2` before rerunning. Windows from an older run (another tile size or raster name) aren't overwritten and would be merged into the new map.


```bash
python Python/detectron2_deploy.py
```

> Note: pretilling operation is not required here because the model internally perform the tiling.

Then, to produce the final shrub masks, use:

```bash
python Python/refine_detections.py \
  --predictions results/deployment/tile_masks/detectron2 \
  --output results/deployment/shrub_map_detectron2.geojson \
  --crs EPSG:25830
```

### SAM 3

The next script create the predictions over the original image. The script also runs the same tiling loop as Detectron2, so no pretiling is required. `shrub` is used as the prompt text:

> Note: empty `results/deployment/tile_masks/sam3/shrub` before rerunning. Windows from an older run (another tile size or raster name) aren't overwritten and would be merged into the new map.

```bash
python Python/sam3_deploy.py
```

Then, to produce the final shrub masks:

```bash
python Python/refine_detections.py \
  --predictions results/deployment/tile_masks/sam3/shrub \
  --output results/deployment/shrub_map_sam3.geojson \
  --crs EPSG:25830
```

## Accuracy assessment

After the deployment phase, each model's shrub map is scored against the manually digitized masks inside the field work plots. Those are the regions of 9 $m^2$ where the field campaigns to digitize the masks were conducted. See `data/encinacorba_annotations.gpkg` `square_rois` and `field_work_masks` layers.

The `field_work_masks` reference layer contains 79 polygons: 58 shrub masks (`shrub == 1`, mean 3.2 $m^2$) and 21 tree masks to assess false positives (`shrub == 0`, mean 15.2 $m^2$). It's exhaustive shrub digitization within each plot, i.e., every shrub inside them was digitized, allowing the computation of the precision-recall metrics. Most reference polygons are whole objects that extend beyond the plots, so no geometry is clipped. Each reference polygon is matched one-to-one against all predictions (IoU ≥ 0.1), and a matched prediction is a true positive wherever it lies. An unmatched prediction is a false positive only if at least half of its area lies inside the plots (`--min-plot-share`, default 0.5), where every shrub was digitized. Outside them, it may be a real shrub that was never digitized.

> Note: The reference layer also contains `esp_code` and `esp_scientific_name` columns with the species code and scientific name per mask.

The following metrics are computed:

- Recall: fraction of ground-reference shrub polygons matched by a prediction.
- Precision: true positives divided by true and false positives.
- Tree commission rate: fraction of ground-reference tree polygons matched by a prediction. It's a diagnostic breakdown of what a chunk of the false positives land on, not a separate test.

To compute the metrics over the ground reference masks, run the following code inside `mrcnn_masks` environmnet:

```bash
python Python/evaluate_deployment_accuracy.py
```

It scores each model's `results/deployment/shrub_map_<model>.geojson` and writes to `results/evaluation/deployment_accuracy/`:

- `summary.csv`: one row per model with the counts (TP, FP, FN, matched trees) and the metrics above.
- `<model>_matches.csv`: whether each reference polygon was matched and its best IoU.

## LiDAR refinement

The model masks come from RGB images only, so a shrub mask can include part of a neighbouring tree crown, or a whole tree can be segmented as a shrub. The LiDAR canopy height model (CHM) gives an independent height measurement that can catch those errors. It is not an input to any model, so the same refinement is applied to the masks of all three. The CHM is built in R at 0.25 m without interpolation, so every valid cell is a direct LiDAR measurement (see [LiDAR processing README](R/README.md)).

For each mask, `Python/lidar_refine_masks.py` takes the CHM cells whose centre falls inside it and assigns one status, in this order:

| Status | Condition | Result |
| ------ | --------- | ------ |
| review | Less than 30% of the cells have a height, or fewer than 5 valid cells | Kept |
| tree | 90th-percentile height (`h_p90`) > 1.7 m and at least 10% of the valid cells above 1.7 m | Rejected |
| clipped | Some cells above 1.7 m, but not enough to reject the mask | Tall cells are cut out and the largest remaining part is kept. Rejected if that part is under 0.25 m² |
| keep | Otherwise | Kept unchanged |

`h_p90` is used instead of the maximum height because it is robust to isolated returns. Close to the ground, CHM heights are within the LiDAR noise, so the CHM only separates trees from shrubs. It does not reject masks dominated by bare ground.

### Threshold calibration

The 1.7 m and 10% thresholds were calibrated on the field reference masks (`field_work_masks`, 58 shrubs and 21 trees). After computing the [height of the annotated masks](R/README.md#height-of-the-annotated-masks), run inside the `mrcnn_masks` environment:

```bash
python Python/lidar_threshold_sweep.py --loo
```

It applies the rule above for every combination of height (1.3 to 1.9 m) and tall fraction (0 to 0.5) and counts how many trees and shrubs end in each status (`threshold_sweep.csv`). The selected combination is the one that rejects the most trees without rejecting or clipping any shrub. Ties go to the higher height, then to the higher fraction. `--loo` repeats the selection leaving out one `square_rois` plot at a time and applies the result to that plot's masks (`threshold_sweep_loo.csv`). This checks that the thresholds don't only fit the masks used to choose them.

### Refine the shrub maps

```bash
python Python/run_lidar_refinement.py
```

It refines each model's `results/deployment/shrub_map_<model>.geojson` with the thresholds above, then scores the result against the field annotations, as in the [accuracy assessment](#accuracy-assessment). The outputs go to `results/lidar_refined/`:

- `model_<model>.gpkg`: retained masks; `model_<model>_rejected.gpkg`: rejected masks.
- `model_<model>_stats.csv`: height statistics and status of every mask.
- `summary.csv`: accuracy metrics after refinement.

## Licenses

This repository combines material under different licenses. Each component keeps its own terms:

| Component | License | Notes |
| --------- | ------- | ----- |
| Code in this repository (`Python/`, `R/`, `bin/shrubs_*`, notebooks) | [MIT](LICENSE) | |
| `bin/mrcnn/` | [MIT, Matterport](bin/mrcnn/LICENSE) | Taken from [Ahmed Gad's Mask-RCNN-TF2](https://github.com/ahmedfgad/Mask-RCNN-TF2), a TensorFlow 2 port of [Matterport's Mask R-CNN](https://github.com/matterport/Mask_RCNN). Modified; see [`bin/mrcnn/README.md`](bin/mrcnn/README.md) for the list of changes. |
| `bin/shrubs_sam3/configs/juniper_frozen_backbone.yaml` | [SAM License](bin/shrubs_sam3/configs/LICENSE) (Meta) | Adapted from the fine-tuning config of the SAM 3 repository, so it is a derivative of SAM Materials and stays under the SAM License, not the MIT license above. |
| Field data (`data/encinacorba_annotations.gpkg`) | CC BY 4.0 | Attribute this repository (see [`CITATION.cff`](CITATION.cff)). |

Third-party software and data that are **not** included in this repository must be obtained under their own licenses:

- **[Detectron2](https://github.com/facebookresearch/detectron2)** (Apache 2.0): installed from its repository, as described above.
- **[SAM 3](https://github.com/facebookresearch/sam3)**: installed with `environment_sam3.yml`. Its checkpoint is gated on Hugging Face and requires accepting the [SAM License](https://github.com/facebookresearch/sam3/blob/main/LICENSE). Fine-tuned SAM 3 checkpoints are derivatives of SAM Materials, so they are not distributed with this repository, and any that are shared later must be shared under that license. Publications using SAM 3 must acknowledge its use.
- **[JuniperMapper](https://zenodo.org/records/10058715)** (CC BY 4.0, Khaldi et al. 2023): download it from Zenodo into `data/JuniperMapper/`. It is not redistributed here. Cite the dataset and [Khaldi et al. 2024](https://doi.org/10.1016/j.jag.2024.104191) when using it or models trained on it.

## Citation

If you use this code, please cite it with the metadata in [`CITATION.cff`](CITATION.cff) (GitHub shows a "Cite this repository" button from it), together with the JuniperMapper dataset and paper.

## Funding

