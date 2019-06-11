import torch.nn as nn
from efficientnet_pytorch import EfficientNet


class Classifier(nn.Module):
    def __init__(self,n_classes):
        super(Classifier, self).__init__()
        self.effnet =  EfficientNet.from_pretrained('efficientnet-b0')
        self.l1 = nn.Linear(1000 , 256)
        self.dropout = nn.Dropout(0.5)
        self.l2 = nn.Linear(256,n_classes) # 6 is number of classes
        self.relu = nn.LeakyReLU()

    def forward(self, input):
        x = self.effnet(input)
        x = x.view(x.size(0),-1)
        x = self.dropout(self.relu(self.l1(x)))
        x = self.l2(x)
        return x