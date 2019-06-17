# Overview

# Approach

In this project, i used several feature extraction networks such as ResNet, EffNet

 

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

# 
