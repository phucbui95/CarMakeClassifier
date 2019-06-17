#!/usr/bin/env bash
mkdir data

cd data

wget http://imagenet.stanford.edu/internal/car196/cars_train.tgz
wget http://imagenet.stanford.edu/internal/car196/cars_test.tgz
wget --no-check-certificate https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz

tar -xvf cars_train.tgz
tar -xvf cars_test.tgz
tar -xvf car_devkit.tgz

