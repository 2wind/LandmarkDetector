import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, models, transforms
import torchvision.transforms.functional as TF

landmark_regex_string = '29@[2479]|30@[34]' # for 18: '29@[1-9]\d?|30@[1-7]'
landmark_number = 6


class LandmarkNetwork(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.model_name='resnet50'
        
        self.model=models.resnet50()
        self.model.conv1=nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc=nn.Linear(self.model.fc.in_features, num_classes)

        
    def forward(self, x):
        x=self.model(x)
        return x