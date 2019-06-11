from options.base_options import BaseOptions
from dataset.cars_dataset import CarDataset
from dataset.data_loaders import DataLoader

from util import convert_batch_to_image, plot_images, EarlyStopping
from utils.lr_finders import LRFinder
import time
from tqdm import tqdm
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from models import Classifier

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(model, criterion, dataloaders, num_epochs=10, lr=0.0003,
                batch_size=32):
    since = time.time()
    model.to(device)
    best_acc = 0.0
    i = 0
    losses = []
    accuracy = []
    earlystop = EarlyStopping(patience=5, verbose=True)
    for epoch in tqdm(range(num_epochs)):
        print('Epoch:', epoch)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        lr = lr * 0.8
        if (epoch % 10 == 0):
            lr = 0.0001

        for phase in ['train', 'val']:
            if phase == ' train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            total = 0
            j = 0
            dataloader = dataloaders[phase]
            for batch_idx, (data, target) in tqdm(
                    enumerate(dataloader), leave=False, total=len(dataloader)):
                data, target = Variable(data), Variable(target)
                data = data.type(torch.cuda.FloatTensor)
                target = target.type(torch.cuda.LongTensor)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                _, preds = torch.max(output, 1)
                running_corrects = running_corrects + torch.sum(
                    preds == target.data)
                running_loss += loss.item() * data.size(0)
                j = j + 1
                if (phase == 'train'):
                    loss.backward()
                    optimizer.step()

                if batch_idx % 300 == 0:
                    print(
                        '{} Epoch: {}  [{}/{} ({:.0f}%)]\tLoss: {:.6f} \tAcc: {:.6f}'.format(
                            phase, epoch, batch_idx * len(data),
                            len(dataloaders[phase].dataset),
                            100. * batch_idx / len(dataloaders[phase])
                            , running_loss / (j * batch_size),
                            running_corrects.double() / (j * batch_size)))
            epoch_acc = running_corrects.double() / (
                        len(dataloaders[phase]) * batch_size)
            epoch_loss = running_loss / (len(dataloaders[phase]) * batch_size)
            if (phase == 'val'):
                earlystop(epoch_loss, model)

            if (phase == 'train'):
                losses.append(epoch_loss)
                accuracy.append(epoch_acc)

        if (earlystop.early_stop):
            print("Early stopping")
            break
        print('{} Accuracy: '.format(phase), epoch_acc.item())
    return losses, accuracy

from torch.utils.data import SubsetRandomSampler
import numpy as np

if __name__ == '__main__':
    option_parser = BaseOptions()
    opt = option_parser.parse()
    dataset = CarDataset('train', opt)

    data_loader = DataLoader(opt, dataset)

    sample_batch, _ = data_loader.next_batch()
    imgs = convert_batch_to_image(sample_batch)
    plot_images(imgs[:4])
    plt.show()

    n_classes = 196

    classifier = Classifier(n_classes)

    indices = list(range(len(dataset)))
    np.random.shuffle(indices)
    test_size = 0.1
    split = int(np.floor(len(indices) * (1 - test_size)))
    train_indices, valid_indices = indices[:split], indices[split:]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(valid_indices)

    batch_size = 32

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               sampler=train_sampler,
                                               num_workers=2)
    valid_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               sampler=valid_sampler,
                                               num_workers=2)

    criterion = nn.CrossEntropyLoss()

    optimizer_ft = optim.Adam(classifier.parameters(), lr=0.0000001)
    lr_finder = LRFinder(classifier, optimizer_ft, criterion, device=device)
    lr_finder.range_test(train_loader, end_lr=1, num_iter=500)
    lr_finder.reset()
    lr_finder.plot()
