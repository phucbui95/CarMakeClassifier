import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import os
import time
import cv2
import torch

import scipy as sp
import scipy.ndimage

import matplotlib.pyplot as plt


def convert_tensor_to_image(tensor):
    img = np.moveaxis(tensor.numpy(), [1, 2], [0, 1])
    img = (img - img.min()) / (img.max() - img.min())
    return img


def convert_batch_to_image(batch_tensor):
    batch_np = batch_tensor.squeeze().numpy()
    if batch_np.ndim == 4:
        img = np.transpose(batch_np, [0, 2, 3, 1])
    else:
        img = np.transpose(batch_np, [0, 1, 2])

    img = (img - img.min()) / (img.max() - img.min())
    return img


def plot_images(images, columns=2):
    w = 10
    h = 10
    fig = plt.figure(figsize=(8, 8))
    n_img = len(images)
    rows = (n_img + columns - 1) // columns
    for i in range(1, n_img + 1):
        img = np.random.randint(10, size=(h, w))
        fig.add_subplot(rows, columns, i)
        plt.imshow(images[i - 1])


def flood_fill(mask, h_max=200):
    input_array = np.copy(mask)
    el = sp.ndimage.generate_binary_structure(2, 2).astype(np.int)
    inside_mask = sp.ndimage.binary_erosion(~np.isnan(input_array),
                                            structure=el)
    output_array = np.copy(input_array)
    output_array[inside_mask] = h_max
    output_old_array = np.copy(input_array)
    output_old_array.fill(0)
    el = sp.ndimage.generate_binary_structure(2, 1).astype(np.int)
    while not np.array_equal(output_old_array, output_array):
        output_old_array = np.copy(output_array)
        output_array = np.maximum(input_array,
                                  sp.ndimage.grey_erosion(output_array,
                                                          size=(3, 3),
                                                          footprint=el))
    return output_array



class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
