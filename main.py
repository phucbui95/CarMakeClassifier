import os.path as osp

import torch
import torchvision.transforms as transforms

from dataset import get_cars_datasets
from dataset.cars_dataset import CarDataset
from dataset.data_loaders import DataLoader
from models import BaseModel
from options.base_options import BaseOptions
from trainer import BaseTrainer, IterationCallback
from utils.auto_augment import ImageNetPolicy
from utils.preprocess import Cutout
from utils.visualization import SummaryWriter

import matplotlib.pyplot as plt
from util import convert_batch_to_image, plot_images

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TensorboardCallback(IterationCallback):
    def __init__(self, opt):
        self.opt = opt
        self.board = SummaryWriter(logdir=osp.join(opt.tensorboard_dir, opt.name))
        self.step = 0

    def update_metrics(self, metrics, phase):
        if phase == 'train':
            self.board.add_scalar('train_loss', metrics['loss'], self.step + 1)
        else:
            self.board.add_scalar('val_loss', metrics['loss'], self.step + 1)
        print("{}\t Loss: {:.6f} \tAcc: {:.6f}".format( phase,
                                                    metrics['loss'],
                                                    metrics['accuracy']))
        self.step += 1



def train(opt):
    trainer = BaseTrainer(opt, device)

    tfs = transforms.Compose([
        transforms.Resize((opt.fine_width, opt.fine_height)),
        ImageNetPolicy(),
        transforms.ToTensor(),
        Cutout(4, opt.fine_width * 0.25),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataloaders = get_cars_datasets(opt, valid_size=0.1, tfs=tfs)
    n_classes = 196
    model = BaseModel(n_classes)
    board = TensorboardCallback(opt)
    trainer.run_train(model, dataloaders, callbacks=[board])

if __name__ == '__main__':
    option_parser = BaseOptions()
    opt = option_parser.parse()

    tfs = transforms.Compose([
        transforms.Resize((opt.fine_width, opt.fine_height)),
        ImageNetPolicy(),
        transforms.ToTensor(),
        Cutout(4, opt.fine_width * 0.25),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = CarDataset('train', opt, image_transformer=None)
    data_loader = DataLoader(opt, dataset)

    batch = data_loader.next_batch()
    sample_batch = batch['data']
    imgs = convert_batch_to_image(sample_batch)
    plot_images(imgs[:4])
    plt.show()

    train(opt)

    # n_classes = 196
    # classifier = Classifier(n_classes)
    #
    # dataloaders = get_cars_datasets(opt, valid_size=0.1)
    # train_loader, valid_loader = dataloaders['train'], dataloaders['valid']
    #
    # criterion = nn.CrossEntropyLoss()

    # optimizer_ft = optim.Adam(classifier.parameters(), lr=0.0000001)
    # lr_finder = LRFinder(classifier, optimizer_ft, criterion, device=device)
    # lr_finder.range_test(train_loader, end_lr=1, num_iter=500)
    # lr_finder.reset()
    # lr_finder.plot()
