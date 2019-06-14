from .cars_dataset import CarDataset
import torch
from torch.utils.data import SubsetRandomSampler
import numpy as np


def get_cars_datasets(opt, valid_size=0.1, tfs=None):
    """ Create data loaders """
    train_set = CarDataset('train', opt, image_transformer=tfs)
    test_set = CarDataset('test', opt, image_transformer=tfs)

    indices = list(range(len(train_set)))
    np.random.shuffle(indices)
    split = int(np.floor(len(indices) * (1 - valid_size)))
    train_indices, valid_indices = indices[:split], indices[split:]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(valid_indices)
    batch_size = opt.batchSize
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               sampler=train_sampler,
                                               num_workers=2)
    valid_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               sampler=valid_sampler,
                                               num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                               sampler=valid_sampler,
                                               num_workers=2)
    return {
        'train' : train_loader,
        'valid' : valid_loader,
        'test' : test_loader
    }


__all__ = [get_cars_datasets]
