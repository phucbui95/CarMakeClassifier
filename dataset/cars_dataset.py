import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import scipy.io
import pandas as pd
import numpy as np

class CarDataset(torch.utils.data.Dataset):

    def __init__(self, data_mode, opt, to_tensor=True, image_transformer=None):
        super().__init__()
        self.opt = opt
        self.data_mode = data_mode
        self.data_folder = opt.dataroot

        data_list_path = os.path.join(self.data_folder,
                                      'devkit',
                                      'cars_{}_annos.mat'.format(self.data_mode))
        self.data_list = self.load_data_list(data_list_path)

        self.fine_width = opt.fine_width
        self.fine_height = opt.fine_height
        self.to_tensor = to_tensor

        self.default_transformer = transforms.Compose([
            transforms.Resize((self.fine_width, self.fine_height)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        if image_transformer is None:
            self.image_transformer = self.default_transformer
        else:
            self.image_transformer = image_transformer

    def name(self):
        return 'Car dataset'

    def load_meta_data(self, file_path=None):
        if file_path is None:
            file_path = os.path.join(self.data_folder, 'devkit/cars_meta.mat')
        cars_meta = scipy.io.loadmat(file_path)
        car_names = [str(n[0]) for n in cars_meta['class_names'][0]]
        car_index = range(1, 1 + len(car_names))
        df_types = pd.DataFrame({'name': car_names, 'index': car_index})
        return df_types

    def load_data_list(self, file_path):
        cars_annos = scipy.io.loadmat(file_path)
        fname = [str(i[-1][0]) for i in cars_annos['annotations'][0]]
        classes = [int(i[-2][0]) for i in cars_annos['annotations'][0]]
        df = pd.DataFrame({'fname': fname, 'classes': classes})
        return df

    def __getitem__(self, index):
        fname = self.data_list.iloc[index]['fname']
        class_ = self.data_list.iloc[index]['classes'] - 1
        fpath = os.path.join(self.data_folder, 'cars_' + self.data_mode, fname)
        im = Image.open(fpath)
        if im.mode != 'RGB':
            im = im.convert('RGB')
        im.thumbnail((self.fine_width, self.fine_height), Image.ANTIALIAS)
        if self.to_tensor:
            im = self.image_transformer(im)
        #             class_ = to_categorical(class_, 196)
        assert class_ >= 0 and class_ < 196
        return (im, class_)

    def __len__(self):
        return len(self.data_list)