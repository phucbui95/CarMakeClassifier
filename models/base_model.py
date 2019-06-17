from efficientnet_pytorch import EfficientNet
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

    def print(self):
        print("===============")
        print(self)
        print("===============")

class RNModel(BaseModel):
    """ Model based-on resnet feature extraction"""
    def __init__(self, n_classes):
        super(BaseModel, self).__init__()
        self.feature_extractor = resnet50(pretrained=True)

        self.l1 = nn.Linear(1000, 256)
        self.dropout = nn.Dropout(0.5)
        self.l2 = nn.Linear(256, n_classes)  # 6 is number of classes
        self.relu = nn.LeakyReLU()

        self._freeze_layers()

        self.create_optimizer()

    def _freeze_layers(self):
        for param in self.feature_extractor.parameters():
            param.requires_grad = True

        for child in list(self.feature_extractor.children())[:-4]:
            for param in child.parameters():
                param.requires_grad = False

    def create_optimizer(self):
        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input):
        x = self.feature_extractor(input)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.l1(x)))
        x = self.l2(x)
        return x

class EfficientNetModel(BaseModel):
    """ Model based-on resnet feature extraction"""
    def __init__(self, n_classes):
        super(BaseModel, self).__init__()
        self.feature_extractor = EfficientNet.from_pretrained('efficientnet-b0')

        self.l1 = nn.Linear(1000, 256)
        self.dropout = nn.Dropout(0.5)
        self.l2 = nn.Linear(256, n_classes)  # 6 is number of classes
        self.relu = nn.LeakyReLU()

        self._freeze_layers()

        self.create_optimizer()

    def _freeze_layers(self):
        for param in self.feature_extractor.parameters():
            param.requires_grad = True

        for child in list(self.feature_extractor.children())[:-4]:
            for param in child.parameters():
                param.requires_grad = False

    def create_optimizer(self):
        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input):
        x = self.feature_extractor(input)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.l1(x)))
        x = self.l2(x)
        return x