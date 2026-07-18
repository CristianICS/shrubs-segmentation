## Introduction

Instance segmentation-based approach to shrub delineation. Apply transfer learning using Mask R-CNN model to obtain shrub masks over RGB images.

## Data

The example tiles to apply the model were collected over a burnt forest (2003) near Encinacorba (Spain), July 17th 2024. They have a spatial resolution of 5 centimeters.

The `data` folder contains the `mask_rcnn_juniper` file, which includes the weights of the trained model using the dataset created by [Khaldi et al. (2023)](https://zenodo.org/records/10058715).

## Installation

### Manual installation using conda

First, make sure you have conda installed from any of the official sources ( [Download Miniconda](https://docs.conda.io/en/latest/miniconda.html)).

Create the conda environment with this line:

```bash
conda create -n mask_rcnn_tf2 python=3.7.3 -y
```

> Note: Python 3.7 is an stable version used to test the implementation.

Set up the environment with the correct dependencies. Run the following line:

```bash
conda create --name mrcnn --file requirements.txt -c conda-forge
```

If the environment were installed correctly, the jupyter notebooks containing the operations to apply and validate the segmentation can be executed. However, the georeferenced masks used in the current project need geopandas and rasterio libraries. In order to avoid version conflicts with the packages used in the `mrcnn` module, a new environment is set with the following line:

```bash
conda create --name masks --file requirements_masks.txt
```

Inside the last environment, the codes `Python\clip_img.py` and `Python\convert_masks.py` can be executed. The first one splits the main orthomosaic into smaller tiles, and the second one transform the spatial files containing the shrub masks into a plain json file in order to retrieve them to train the Mask RCNN model ???????????????.

## Usage

First, download the current repository.

```bash
git clone https://github.com/
```

The `bin` folder contains the `mrcnn` module from [Ahmed Gad's Mask-RCNN-TF2 repository](https://github.com/ahmedfgad/Mask-RCNN-TF2). It is used to apply the model Mask R-CNN to the dataset.

> The original [Matterport Mask RCNN repo](https://github.com/matterport/Mask_RCNN) is built with TensorFlow 1.15 which is now no longer supported and it has been updated by TensorFlow 2.

## Funding

It was produced through a Master Thesis, with data (mencionar proyecto PAF y beca DGA)



Directorio principal, contiene los cuadernos para ejecutar las principales operaciones del entrenamiento con Mask RCNN.

./
|   shrubs_exploration.ipynb ---> Calcular métricas de acierto del modelo.
|   shrubs_map.ipynb         ---> Generar un mapa de clasificación de matorrales
|   shrubs_train.ipynb       ---> Entrenamiento del modelos con los datos de https://zenodo.org/records/10058715

La carpeta "data" contiene los archivos necesarios para aplicar los procesos de clasificación.

+---data
|   |   encinacorba_cloud6_groundcls.las ---> Nube de puntos clasificada (código R/create_dsm.R)
|   |   encinacorba_orto.tif             ---> Ortofoto 0.5 cm/px de encinacorba
|   |   mask_rcnn_coco.h5                ---> Pesos del modelo Mask RCNN utilizado para realizar "transfer learning"
|   |   
|   +---encinacorba_tiles                ---> Ortofoto dividida en pequeños tiles con los que se aplicará el modelo.
|   |       tile_0_0.tif
|   |       ...
|   |
|   +---espectros                        ---> Firmas espectrales obtenidas con el radiómetro ASD FieldSpec
|   |       PLOT10_spec181.asd
|   |       ...
|   |       
|   +---espectros_metadata
|   |       metadatos.csv
|   |       README.txt
|   |       spectre_names.csv
|   |       spec_points.gpkg
|   |       spec_squares.gpkg
|   |       veg_codes.csv
|   |       
|   \---JuniperMapper                    ---> Imágenes con las que se realiza "transfer learning"
|       +---Field_Work_Data
|       |               
|       \---Photo_Interpretation_Data

En el directorio "Python" se encuentran los código necesario para procesar las imágenes RGB. El directorio "mrcnn" contiene las funciones necesarias para ejecutar el modelo de segmentación. Se han descargado del repositorio de github https://github.com/ahmedfgad/Mask-RCNN-TF2

+---Python
|   |   clip_img.py                     ---> Separa la imagen principal en teselas más pequeñas.
|   |   convert_masks.py                ---> Transforma las máscaras de matorral (SHP) a texto plano para que puedan ser incluidas en el modelo.
|   |   environment_masks.txt
|   |   environment_mrcnn.txt
|   |   environment_mrcnn_nobuilds.txt
|   |   masks.py
|   |   README.md
|   |   utils.py
|   |   
|   +---mrcnn
|   |       config.py
|   |       model.py
|   |       model_temp.py
|   |       parallel_model.py
|   |       utils.py
|   |       visualize.py
|   |       __init__.py

Dentro de la carpeta "R" se encuentran los códigos para obtener el DSM y el DEM, además de los buffers cuadrados de 5m alrededor de cada punto.

+---R
|       create_dsm.R
|       create_square_rois.R
|       preprocess_spectres.R
|       spectre_utils.R

Por último, en la carpeta de resultados, se guarda la siguiente información.

.
|       
\---results
    |   encinacorba_cloud6_normalize.las       ---> Nube de puntos normalizada (alturas relativas sobre 0)
    |   encinacorba_dem.tif                    ---> Modelo Digital de Elevaciones
    |   encinacorba_dsm.tif                    ---> Modelo Digital de Superficies
    |   
    +---logs                                   ---> Carpeta con los pesos del modelo entrenado con 20 epochs.
    |   \---shrubs_cfg
    |       |   mask_rcnn_shrubs_cfg_0001.h5
    |       |   ...
    |       |   
    |       +---train
    |       \---validation
    |               
    \---shrub_map_mrcnn                        ---> Aplicación del modelo MRCNN sobre la ortofoto de Encinacorba.
        |   shrubs_map.vrt                     ---> Mapa de matorrales (1) sobre la zona de estudio obtenido aplicando el modelo Mask RCNN.
        |   
        +---png_tiles                          ---> Máscaras resultado de la aplicación del modelo (1 == matorral). No tienen coordenadas.
        |       tile_0_2048_mask.png
        |       ...
        |       
        \---tif_tiles                          ---> Mismas imágenes que "png_tiles", pero con coordenadas. Se calculan aplicando el código "Python/convert_masks"
                tile_0_0_mask.tif
                ...