import torch.nn as nn
import torch.nn.functional as F


class BKVGG12(nn.Module):

    def __init__(self, num_classes=7):
        super(BKVGG12, self).__init__()
        self.features = nn.Sequential(

            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  #conv3 32
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(negative_slope=0.05, inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1), #conv3 32
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(negative_slope=0.05, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # conv3 64
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(negative_slope=0.05, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # conv3 64
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(negative_slope=0.05, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # conv3 128
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(negative_slope=0.05, inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  # conv3 128
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(negative_slope=0.05, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # conv3 256
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(negative_slope=0.05, inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # conv3 256
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(negative_slope=0.05, inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # conv3 256
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(negative_slope=0.05, inplace=True),

        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 5 * 5, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes) 

        )

    def forward(self, x):
        x = self.features(x)

        x = x.view(x.size(0), -1)

        x = self.classifier(x)
        return F.log_softmax(x)