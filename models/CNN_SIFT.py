import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class CNN_SIFT(nn.Module):

    def __init__(self, num_classes=7,use_cuda=True):
        super(CNN_SIFT, self).__init__()
        self.use_cuda=use_cuda
        self.features = nn.Sequential(

            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  #conv3 32
            nn.LeakyReLU(negative_slope=0.05, inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1), #conv3 32
            nn.LeakyReLU(negative_slope=0.05, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.1),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # conv3 64
            nn.LeakyReLU(negative_slope=0.05, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # conv3 64
            nn.LeakyReLU(negative_slope=0.05, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.1),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # conv3 128
            nn.LeakyReLU(negative_slope=0.05, inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  # conv3 128
            nn.LeakyReLU(negative_slope=0.05, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(128 * 5 * 5 + 16*128, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes)
        )

    def forward(self, x):
        x = self.features(x)

        x = x.view(x.size(0), -1)
        features=torch.from_numpy(np.zeros((50,16*128))) #this will be replaced by a flattened array containing dense SIFT features for each image
        if self.use_cuda:
            features=features.cuda()

        features=Variable(features, requires_grad = False).float()
        print(type(x))
        print(type(features))
        x=torch.cat((x,features),1)
        print x.size(1)

        x = self.classifier(x)
        return F.log_softmax(x)