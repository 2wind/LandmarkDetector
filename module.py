import torch.nn as nn
from torchvision import models

# for 6 landmarks:  '29@[2479]|30@[34]' 
# for 18 landmarks: '29@[1-9]\d?|30@[1-7]'
landmark_regex_string = '29@[2479]|30@[34]'
landmark_number = 6


class LandmarkNetwork(nn.Module):
    '''
    class LandmarkNetwork(nn.Module)
    Landmark detection module based on resnet50.

    1st Convolution layer is changed to 1 resolution, to use grayscale image(instead of RGB).
    Final Fully connected layer is changed to 2048 --> num_classes Linear function, 
    to detect num_classes=landmark_number * 2 (x and y) numbers.

    Landmarks coordinates are normalized between -0.5 and 0.5.
    Input image is normalized into 224x224x1 grayscale image, converted into tensor.

    '''
    def __init__(self,num_classes):
        super().__init__()
        self.model_name='resnet50'
        
        self.model=models.resnet50()
        self.model.conv1=nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc=nn.Linear(self.model.fc.in_features, num_classes)

        
    def forward(self, x):
        x=self.model(x)
        return x