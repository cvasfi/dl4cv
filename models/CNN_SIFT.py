import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2
import random

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
        sift_features=[]
        batchsize=x.size()[0]
        imgs = x.data.cpu().numpy()

        x = self.features(x)
        x = x.view(x.size(0), -1)

        for imidx in range(batchsize):
            img=np.squeeze(imgs[imidx,:],0)

            #def denormalize(n):
            #    return int(n*255)
            #denormalize=np.vectorize(denormalize)
            ##img=denormalize(img)

            img = np.array(img * 255, dtype=np.uint8)
            img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 0)  #numpy to opencv conversion, may be wrong
            #cv2.imwrite('test/'+str(random.randint(0,100))+".png",img)
            sift = cv2.xfeatures2d.SIFT_create()

            kp=[]
            for r in range(0,48,12):
                for c in range(0,48,12):
                    kp.append(cv2.KeyPoint(r,c,12))

            sift_features.append(np.array(sift.compute(img, kp)[1]).flatten())

        sift_features=np.vstack(tuple(sift_features))
        sift_features = torch.from_numpy(sift_features)

        if self.use_cuda:
            sift_features=sift_features.cuda()

        sift_features=Variable(sift_features, requires_grad = False).float()
        x=torch.cat((x,sift_features),1)

        x = self.classifier(x)
        return F.log_softmax(x)