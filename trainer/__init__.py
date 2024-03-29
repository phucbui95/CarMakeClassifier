import time

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt

from abc import abstractmethod, ABC
from util import EarlyStopping

def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

if not isnotebook():
    from tqdm import tqdm
else:
    from tqdm import tqdm_notebook as tqdm

class IterationCallback(ABC):
    @abstractmethod
    def update_metrics(self, metrics, phase):
        pass

class MetricsTracker:
    def __init__(self, dataloaders=None):
        self.since = time.time()
        self.best_acc = 0.0
        self.i = 0
        self.losses = []
        self.accuracy = []

        self.val_losses = []
        self.val_accuracy = []
        self.phase = 'train'
        self.epoch = 0

        self.dataloaders = dataloaders
        self.batch_idx = 0

    def update_metrics(self, metrics, phase):
        pass

    def end_valid_iteration(self, epoch_loss, epoch_acc):
        self.val_losses.append(epoch_loss)
        self.val_accuracy.append(epoch_acc)
        self.phase = 'train'
        self.batch_idx += 1

    def end_train_iteration(self, epoch_loss, epoch_acc):
        self.losses.append(epoch_loss)
        self.accuracy.append(epoch_acc)
        self.phase = 'valid'
        self.batch_idx += 1

    def end_epoch(self):
        self.epoch += 1
        self.batch_idx = 0

    # def print_report(self):
    #     if self.batch_idx % 100 == 0:
    #         print(
    #             '{} Epoch: {}  [{}/{} ({:.0f}%)]\tLoss: {:.6f} \tAcc: {:.6f}'.format(
    #                 self.phase,
    #                 self.epoch,
    #                 self.batch_idx * batch_size,
    #                 len(self.dataloaders[self.phase].dataset),
    #                 100. * self.batch_idx / len(self.dataloaders[self.phase]),
    #                 self.running_loss / (j * batch_size),
    #                 self.running_corrects.double() / (j * batch_size)))

class BaseTrainer(ABC):
    def __init__(self, opt, device):
        self.opt = opt
        self.device = device

    def run_one_step(self, model, batch_data, phase):
        data, target = batch_data['data'], batch_data['class']
        data, target = Variable(data), Variable(target)

        if self.device.type != 'cpu':
            data = data.type(torch.cuda.FloatTensor)
            target = target.type(torch.cuda.LongTensor)
        else:
            data = data.type(torch.FloatTensor)
            target = target.type(torch.LongTensor)

        model.optimizer.zero_grad()
        output = model(data)
        loss = model.criterion(output, target)
        _, preds = torch.max(output, 1)

        running_corrects = torch.sum(preds == target.data)
        running_loss = loss.item() * data.size(0)

        if phase == 'train':
            loss.backward()
            model.optimizer.step()

        return running_corrects, running_loss

    def run_train(self, model, dataloaders, num_epochs=10, callbacks=None):
        if callbacks is None:
            callbacks = []
        model.to(self.device)
        earlystop = EarlyStopping(patience=20,
                                  verbose=True,
                                  checkpoint='checkpoint_effnet_large')
        tracker = MetricsTracker()
        batch_size = self.opt.batchSize

        for epoch in tqdm(range(num_epochs)):
            print('Epoch:', epoch)
            for phase in ['train', 'valid']:
                if phase == ' train':
                    model.train()
                else:
                    model.eval()
                running_loss = 0.0
                running_corrects = 0

                dataloader = iter(dataloaders[phase])
                for batch_idx, batch_data in tqdm(
                        enumerate(dataloader), leave=False,
                        total=len(dataloader)):

                    data, target = batch_data['data'], batch_data['class']
                    running_corrects_, running_loss_ = self.run_one_step(model,
                                                                         batch_data,
                                                                         phase)
                    accuracy = float(running_corrects) / ((batch_idx + 1) * batch_size)
                    metrics = dict(accuracy=accuracy,
                                   loss=running_loss_ / data.size(0))
                    if batch_idx % self.opt.display_count == 0:
                        for cb in callbacks:
                            cb.update_metrics(metrics, phase)

                    running_corrects += running_corrects_
                    running_loss += running_loss_

                epoch_acc = running_corrects.double() / (len(dataloaders[phase]) * batch_size)
                epoch_loss = running_loss / (
                            len(dataloaders[phase]) * batch_size)

                if phase == 'valid':
                    tracker.end_valid_iteration(epoch_loss, epoch_acc.item())
                    earlystop(epoch_loss, model)

                if phase == 'train':
                    tracker.end_train_iteration(epoch_loss, epoch_acc.item())

            if (earlystop.early_stop):
                print("Early stopping")
                break
            print('{} Accuracy: '.format(phase), epoch_acc.item())
            tracker.end_epoch()




