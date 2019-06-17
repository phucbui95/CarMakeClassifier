# Overview
       The Cars dataset contains 16,185 images of 196 classes of cars. The data is split into 8,144 training images and 8,041 testing images, where each class has been split roughly in a 50-50 split. Classes are typically at the level of Make, Model, Year, e.g. 2012 Tesla Model S or 2012 BMW M3 coupe.
       
       
# Approach

In this project, i used several feature extraction networks such as ResNet, VGG, EffNet

# Installation

This project was written and test on `python3.6` and `pytorchv1`

Please install library in requirements.txt before continue


## Download dataset
This script will download and unzip Standford Cars datasets into data folder

```bash
./download_dataset.sh
```

## Preprocessing

```bash
python crop_image.py
```

## Training
```bash
python train.py
```

## Inference
```bash
python test.py
```

By default, the output will be written to `outputs` folder.
