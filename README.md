# Detection of thinning necessity
Detection of the necessity of thinnings
We apply deep convolutional neural networks for semantic sementation to detect the need for thinning in the forest from remmote sensing data.




###########################################################################

# create_dataset
data preperation and concatination into data set

## Create input data tiles
prepare_data_ortho.ipynb
prepare_data_dsm.ipynb
prepare_data_dtm.ipynb
prepare_data_slope.ipynb

preperation of the data for concationation into one data set (see Create data set).
in particular the raster files are clipt into tiles defined by the vector file (.shp).
in the case of dsm, dtm and slope the created files need to be aigned since they 
have diffrent spatial resolutions.

## Create ground truth tiles
create_ground_truth.ipynb

Created ground truth raster files (.tif) out of a vector file (.shp)

### add ground truth types
defines the ground truths for Base, UR12 and UR1 (see disertation).


## Create data set
create_dataset_mult_gt.ipynb

### data set creation (hdf5-file: .h5)
Creates the data set as a single hdf5 file. The data must by prepared in order 
to allow the script run smothly.
* set input data directory containing the preprocessed data
* inside this directoy the directories must have the exact names as defined in 
the dictionary ´data_dict
* all files in this dictonaries must have a specific prefix

directory: "xxx", tile number: "nnnnnn", file name: "tile_xxx_nnnnnn.tif"
for example
directory: "dtm", tile number: "122277", file name: "tile_dtm_122277.tif"

### calculation of stats
calculation of mean and standard deviation and addition to data set (.h5)


## Create extended data set (fliped)
dataset_flip.ipynb
indices of train, validation and test sets must be prepared as numpy arrays (.npy)

###########################################################################

# deep_learing
the three folders contain the code for training and evaluation of the models.
each of the folder has the same structure.

deeplabv3plus - DeepLabv3+
fcdensenet - FC-FCDenseNet
unet - UNet

datasets - loading and preparing data for the DCNNs
log - for logging the progress of the training
models - contains the implementation of the DCNNs
utils - additional code, e.g. saving weighs, visulising data, ...
weights - for storing the trained weights (every epoch)

train.ipynb - for training models
test.ipynb - for testing models

## execution
it is just necessary to run train.ipynb for training and test.ipynb for testing

###########################################################################

# prediction
for prediction of the necessity of thinnings. creation of verctor (.shp) data
out of the predicted raster tiles from the DCNNs.

the folder structure is the same as in # deep_learing

## create predictions
predict_data_512.ipynb

creates predictions as raster tiles.

## create vector data
predict_data_rasterize.ipynb

concatonates the tiles created by predict_data_512.ipynb and vectorises the data. 
