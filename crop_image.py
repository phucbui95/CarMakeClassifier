import os
import os.path as osp

import pandas as pd
from PIL import Image
from scipy.io import loadmat
from tqdm import tqdm as tqdm
from options.base_options import  BaseOptions

def load_meta(car_annos, datamode):
    annotations = car_annos["annotations"][0, :]
    nclasses = len(car_meta["class_names"][0])
    class_names = dict(
        zip(range(1, nclasses), [c[0] for c in car_meta["class_names"][0]]))

    labelled_images = {}
    dataset = []
    for i, arr in enumerate(annotations):
        # the last entry in the row is the image name
        # The rest is the data, first bbox, then classid
        if datamode == 'train':
            dataset.append([y[0][0] for y in arr][0:5] + [arr[5][0]])
        else:
            dataset.append([y[0][0] for y in arr][0:4] + [0] + [arr[4][0]])
    # Convert to a DataFrame, and specify the column names
    temp_df = pd.DataFrame(dataset,
                           columns=['BBOX_X1', 'BBOX_Y1', 'BBOX_X2', 'BBOX_Y2',
                                    'ClassID', 'filename'])

    temp_df = temp_df.assign(ClassName=temp_df.ClassID.map(dict(class_names)))
    temp_df.columns = ['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'class_id',
                       'filename', 'class_name']
    return temp_df



def crop_image(opt, df, output_dir, datamode):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in tqdm(range(0, len(df))):
        filename = df.iloc[i]['filename']
        x1 = df.iloc[i]['bbox_x1']
        y1 = df.iloc[i]['bbox_y1']
        x2 = df.iloc[i]['bbox_x2']
        y2 = df.iloc[i]['bbox_y2']

        src = Image.open(osp.join(opt.dataroot, 'cars_' + datamode, filename))
        area = [x1, y1, x2, y2]
        cropped_img = src.crop(area)

        cropped_img.save(os.path.join(output_dir, filename))

if __name__ == '__main__':
    option_parser = BaseOptions()
    opt = option_parser.parse()
    devkit_path = 'data/devkit/'
    cars_train = 'data/cars_train/'
    cars_test = 'data/cars_test/'

    car_meta = loadmat(devkit_path + 'cars_meta.mat')
    cars_train_annos = loadmat(devkit_path + 'cars_train_annos.mat')
    cars_test_annos = loadmat(devkit_path + 'cars_test_annos.mat')

    train_df = load_meta(cars_train_annos, 'train')
    test_df = load_meta(cars_test_annos, 'test')

    crop_image(opt, train_df, osp.join(opt.dataroot, 'cars_train' + '_cropped'), 'train')
    crop_image(opt, test_df, osp.join(opt.dataroot, 'cars_test' + '_cropped'), 'test')
