#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      https://github.com/kazuto1011
# Created:  2017-04-20


from __future__ import print_function
import argparse
import os.path as osp
from datetime import datetime
import tensorboard_logger as logger
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.autograd import Variable
from models import VGG, ResNetCifar10, BKVGG12, CNN_SIFT
from facedata import FaceData
import numpy as np
import os
import time

parser = argparse.ArgumentParser(
    description='Place Categorization on Sparse MPO')
parser.add_argument('--batch-size', type=int, default=256, metavar='N')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N')
parser.add_argument('--epochs', type=int, default=500, metavar='N')
parser.add_argument('--no-cuda', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=1, metavar='S')
parser.add_argument('--log-interval', type=int, default=50, metavar='N')
parser.add_argument('--save-interval', type=int, default=50, metavar='N')
parser.add_argument('--fold', type=int, default=0)
parser.add_argument('--optimizer', type=str, default='adam')
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--lr-decay', type=float, default=0.1)
parser.add_argument('--lr-decay-after', type=float, default=250)
parser.add_argument('--weight-decay', type=float, default=1e-4)
parser.add_argument('--tensorboard', action='store_true')
parser.add_argument('--dropout-rate', type=float, default=0.5)
parser.add_argument('--write-csv', type=bool, default=False)
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

print('Arguments')
for arg in vars(args):
    print('{0:20s}: {1}'.format(arg.rjust(20), getattr(args, arg)))


class average_meter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

train_history = list()
test_history = list()

def train(epoch, model, optimizer, loader):
    print("Learning rate " + str(args.lr))
    losses = average_meter()
    accuracy = average_meter()

    model.train()
    for batch_idx, (data, target) in enumerate(loader):
        #data=data.type(torch.FloatTensor)
        #target=target.type(torch.LongTensor)
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        #print("Input Type")
        #print(type(data))
        data, target = Variable(data).float(), Variable(target)


        output = model(data)
        loss = F.nll_loss(output, target)
        losses.update(loss.data[0], data.size(0))

        pred = output.data.max(1)[1]
        prec = pred.eq(target.data).cpu().sum()
        accuracy.update(float(prec) / data.size(0), data.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {}\t'
                  'Batch: [{:5d}/{:5d} ({:3.0f}%)]\t'                     
                  'Loss: {:.6f}'.format(
                      epoch, batch_idx * len(data), len(loader.dataset),
                      100. * batch_idx / len(loader), losses.val))
            print('Training accuracy:', accuracy.val )
    if args.tensorboard:
        logger.log_value('train_loss', losses.avg, epoch)
        logger.log_value('train_accuracy', accuracy.avg, epoch)

    train_history.append((epoch, losses.avg, accuracy.avg))
    return accuracy.avg


def vldtn(epoch, model, optimizer, loader):
    losses = average_meter()
    accuracy = average_meter()

    model.eval()
    for data, target in loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True).float(), Variable(
            target, volatile=True)

        output = model(data)
        loss = F.nll_loss(output, target)
        losses.update(loss.data[0], data.size(0))

        pred = output.data.max(1)[1]
        prec = pred.eq(target.data).cpu().sum()
        accuracy.update(float(prec) / data.size(0), data.size(0))

    print('\nTest: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        losses.avg, int(accuracy.sum), len(loader.dataset), 100. * accuracy.avg))

    if args.tensorboard:
        logger.log_value('test_loss', losses.avg, epoch)
        logger.log_value('test_accuracy', accuracy.avg, epoch)
    test_history.append((epoch, losses.avg, accuracy.avg))

    return accuracy.avg

def test(model, loader):
    losses = average_meter()
    accuracy = average_meter()

    model.eval()
    for data, target in loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True).float(), Variable(
            target, volatile=True)

        bs, ncrops, c, h, w = data.size()
        temp_output = model(data.view(-1, c, h, w))
        output = temp_output.view(bs, ncrops, -1).mean(1)

        loss = F.nll_loss(output, target)
        losses.update(loss.data[0], data.size(0))

        pred = output.data.max(1)[1]
        prec = pred.eq(target.data).cpu().sum()
        accuracy.update(float(prec) / data.size(0), data.size(0))

    print('\nTest: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        losses.avg, int(accuracy.sum), len(loader.dataset), 100. * accuracy.avg))


    return accuracy.avg


def main():

    if args.tensorboard:
        logger.configure(
            osp.join('log',
                     'model_{}'.format(args.model),
                     'batchsize_{}'.format(args.batch_size),
                     'optimizer_{}'.format(args.optimizer),
                     'lr_{}'.format(args.lr),
                     'weightdecay_{}'.format(args.weight_decay),
                     datetime.now().isoformat()), flush_secs=1)

    # data augmentation
    transform_train = transforms.Compose([transforms.RandomHorizontalFlip(),
                                     transforms.RandomRotation(45),
                                     transforms.RandomResizedCrop(42, scale=(0.875, 1.125), ratio=(1.0, 1.0)),
                                     #transforms.RandomCrop(42)
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.507395516207, ),(0.255128989415, ))
                                    ])

    validation_transform = transforms.Compose([transforms.Resize(42),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.507395516207,), (0.255128989415,))
                                     ])

    test_transform = transforms.Compose([transforms.TenCrop(42),
                                    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                    transforms.Lambda(lambda crops: torch.stack([transforms.Normalize((0.507395516207,), (0.255128989415,))(crop) for crop in crops])),
                                    ])

    trn_dataset = FaceData(dataset_csv="data/fer2013.csv", dataset_type='Training', transform=transform_train)
    val_dataset = FaceData(dataset_csv="data/fer2013.csv", dataset_type='PublicTest', transform=validation_transform)
    tst_dataset = FaceData(dataset_csv="data/fer2013.csv", dataset_type='PrivateTest', transform=test_transform)

    train_loader = torch.utils.data.DataLoader(trn_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    valid_loader  = torch.utils.data.DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=4)
    test1_loader = torch.utils.data.DataLoader(tst_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=4)

    model = {
        'vgg': VGG(),
        'resnet20': ResNetCifar10(n_block=3),
        'resnet32': ResNetCifar10(n_block=4),
        'resnet44': ResNetCifar10(n_block=5),
        'resnet56': ResNetCifar10(n_block=6),
        'resnet110': ResNetCifar10(n_block=18),
        'bkvgg12': BKVGG12(7, dropout_rate=args.dropout_rate),
        "cnn_sift": CNN_SIFT(7, args.cuda)
    }.get(args.model)

    if args.cuda:
        model.cuda()

    optimizer = {
        'adam': optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay),
        'momentum_sgd': optim.SGD(model.parameters(), args.lr, momentum=0.9, weight_decay=args.weight_decay),
        'nesterov_sgd': optim.SGD(model.parameters(), args.lr, momentum=0.9, weight_decay=args.weight_decay, nesterov=True),
    }.get(args.optimizer)

    results = []
    best_model = model
    best_accuray = 0.0

    for epoch in range(1, args.epochs + 1):

        lr = args.lr * (0.1 ** (epoch //  args.lr_decay_after))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        train_accuracy = train(epoch, model, optimizer, train_loader)
        val_accuracy   = vldtn(epoch, model, optimizer, valid_loader)

        results.append((model, train_accuracy, val_accuracy))


        if epoch % args.save_interval == 0:
            torch.save(model.state_dict(),
                       'log/{}_epoch{}.model'.format(args.model, epoch))

        if best_accuray < val_accuracy:
            best_model   = model
            best_accuray = val_accuracy

    if(args.write_csv):
        key = time.time()
        directory = ('training%f').format(str(key))
        os.makedirs(key)
        np.savetxt(directory + "/train_history.csv", train_history, delimiter=",", header="epoch,loss, accuracy", comments="")
        np.savetxt(directory + "/test_history.csv", test_history, delimiter=",", header="epoch,loss, accuracy",   comments="")

    print ("The best model has an accuracy of " + str(best_accuray))

    torch.save(best_model.state_dict(), 'best.model')

    #Test on Private Test
    test(best_model, test1_loader)


if __name__ == '__main__':
    main()
