# Detection of thinning necessity
Detection of the necessity of thinnings
We apply deep convolutional neural networks for semantic sementation to detect the need for thinning in the forest from remmote sensing data.

***

## 1. If you don't have it - Install conda
1. **Check you don't already have conda installed!**
    1. `which conda`
    1. **if you already have it installed, skip ahead to Create an Environment**
    1. It doesn't matter if you have miniconda3, or anaconda3 installed (it does not even matter if it is version 2).
1. If you don't have conda, download the latest version of miniconda3
    1. `cd ~/Downloads` (you can make a Downloads folder if you don't have one)
    1. Download the installer (we prefer to use miniconda since it carries less baggage), depending on your system (you can check links [here](https://conda.io/miniconda.html)):
        * Linux: `wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh`
        * Mac: `wget https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh` or ```curl -LOk https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh```
        * Or just simply download from [the site](https://conda.io/miniconda.html)
1. Install miniconda3 *with default settings*
    1. `bash Miniconda3-latest-Linux-x86_64.sh`
    1. Follow the prompt - **type `yes` and hit `enter` to accept all default
    settings when asked**
1. Close Terminal and reopen
1. Try executing `conda -h`. If it works, you can delete the installer
```rm ~/Downloads/Miniconda3-latest-Linux-x86_64.sh```

## 3a. Create an environment for geo
1. Update conda: `conda update conda`
1. Create the environment. Call it geo and install python 3 (*hence the name*):
```conda create -n geo```
1. Install packages: `conda install xxx`
* numpy
* geopandas
* osgeo
* h5py

## 3b. Create an environment for pytorch
1. Create the environment. Call it pytorch and install python 3 (*hence the name*):
```conda create -n pytorch```
1. Install packages:
`conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch`
`conda install pandas`
`conda install h5py`

***

## create_dataset
data preperation and concatination into data set

#### Create input data tiles
- `prepare_data_ortho.ipynb`
- `prepare_data_dsm.ipynb`
- `prepare_data_dtm.ipynb`
- `prepare_data_slope.ipynb`

preperation of the data for concationation into one data set (see Create data set).
in particular the raster files are clipt into tiles defined by the vector file (.shp).
in the case of dsm, dtm and slope the created files need to be aigned since they 
have diffrent spatial resolutions.

#### Create ground truth tiles
`create_ground_truth.ipynb`

Created ground truth raster files (.tif) out of a vector file (.shp)

*add ground truth types* defines the ground truths for Base, UR12 and UR1 (see disertation).


#### Create data set
`create_dataset_mult_gt.ipynb`

#### data set creation (hdf5-file: .h5)
Creates the data set as a single hdf5 file. The data must by prepared in order 
to allow the script run smothly.
* set input data directory containing the preprocessed data
* inside this directoy the directories must have the exact names as defined in 
the dictionary ´data_dict
* all files in this dictonaries must have a specific prefix

directory: `xxx`, tile number: `nnnnnn`, file name: `tile_xxx_nnnnnn.tif`

directory: `dtm`, tile number: `122277`, file name: `tile_dtm_122277.tif`

*calculation of stats* calculation of mean and standard deviation and addition to data set (.h5)


#### Create extended data set (fliped)
`dataset_flip.ipynb`
indices of train, validation and test sets must be prepared as numpy arrays (.npy)

***

## deep_learing
the three folders contain the code for training and evaluation of the models.
each of the folder has the same structure.

deeplabv3plus - DeepLabv3+
fcdensenet - FC-FCDenseNet
unet - UNet

- `datasets` - loading and preparing data for the DCNNs
- `log` - for logging the progress of the training
- `models` - contains the implementation of the DCNNs
- `utils` - additional code, e.g. saving weighs, visulising data, ...
- `weights` - for storing the trained weights (every epoch)

`train.ipynb` - for training models
`test.ipynb` - for testing models

#### execution
it is just necessary to run train.ipynb for training and test.ipynb for testing

***

## prediction
for prediction of the necessity of thinnings. creation of verctor (.shp) data
out of the predicted raster tiles from the DCNNs.

the folder structure is the same as in # deep_learing

#### create predictions
`predict_data_512.ipynb`

creates predictions as raster tiles.

#### create vector data
`predict_data_rasterize.ipynb`

concatonates the tiles created by `predict_data_512.ipynb` and vectorises the data. 
