import torch.nn as nn
from torchvision.models import resnet50
import torch.optim as optim

from abc import ABC, abstractmethod

class BaseModel(nn.Module):

    def __init__(self, opt):
        super().__init__()
        self.opt = opt

    @abstractmethod
    def create_optimizers(self, opt):
        pass

