## Tensorflow VGG16

Using a trained VGG16 from: https://github.com/machrisaa/tensorflow-vgg

>To use the VGG networks, the npy files for [VGG16 NPY](https://mega.nz/#!YU1FWJrA!O1ywiCS2IiOlUCtCpI6HTJOMrneN-Qdv3ywQP5poecM) has to be downloaded. In order to load weights

## Installation

* Anaconda [here](https://www.anaconda.com/download/#macos)
```
bash Anaconda3-4.4.0-Linux-x86_64.sh
source ~/.bashrc
```
* Tensorflow
```
conda create -n tensorflow
source activate tensorflow
(tensorflow) pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.2.1-cp36-cp36m-linux_x86_64.whl
```
* CUDA 8.0 and cuDNN 5.1
Add the following to ~/.bashrc
```
module add openmind/cuda/8.0
module add openmind/cudnn/8.0-5.1
```



## Usage
To run this script, predict.py, Tensorflow must be installed. 
The following parameters must be changed:
```
directory # where images can be found
img_paths # paths of images with labels, each line of the txt file: name_of_image.ext label
layer_path # directory where pool layers are stored, not including poolX.npy
pca_layer_path # directory where pca layers are stored
pool_layer_path # directory where pool layers are stored
use_pca # True if you wish to use the PCA function
cross_val
num_trials # number of times to run svm
```
Ensure that the correct pooling layer is uncommented and the directories all align
