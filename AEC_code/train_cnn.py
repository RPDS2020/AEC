#This code is for training ANN

import torch
import random
import numpy as np
import os
import copy
import time
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100
from data.autoaugment import CIFAR10Policy, Cutout
from models.vgg import VGG
from models.new_convert_code_1 import SpikeModel,SpikeModule
from models.fold_bn import search_fold_and_remove_bn
from find_data_distribute import find_activation_percentile
#删除了workers=4
def build_data(dpath: str, batch_size=128, cutout=False,  use_cifar10=True, auto_aug=False):

    aug = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    if auto_aug:
        aug.append(CIFAR10Policy())

    aug.append(transforms.ToTensor())

    if cutout:
        aug.append(Cutout(n_holes=1, length=16))

    if use_cifar10:
        aug.append(
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
        transform_train = transforms.Compose(aug)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = CIFAR10(root=dpath, train=True, download=True, transform=transform_train)
        val_dataset = CIFAR10(root=dpath, train=False, download=True, transform=transform_test)

    else:
        aug.append(
            transforms.Normalize(
                (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        )
        transform_train = transforms.Compose(aug)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        train_dataset = CIFAR100(root=dpath, train=True, download=True, transform=transform_train)
        val_dataset = CIFAR100(root=dpath, train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                               pin_memory=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False,  pin_memory=True)

    return train_loader, val_loader

#网络参数默认值
batch_size = 32
learning_rate = 1e-2
epochs = 300
dataset = 'CIFAR10'
usebn = True
wd = 5e-4
seed = 1000

dpath = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(
    __file__)), '.','data_set'))


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = epochs
    use_cifar10 = True
    train_loader, test_loader = build_data(dpath=dpath, cutout=True, use_cifar10=use_cifar10, auto_aug=True)
    best_acc = 0
    best_epoch = 0
    use_bn = usebn
    wd = 5e-4 if use_bn else 1e-4
    bn_name = 'wBN' if use_bn else 'woBN'
    model_save_name =  'ann.pth'


    ann = VGG('VGG16', use_bn=use_bn, num_class=10 if use_cifar10 else 100)
    ann.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    # build optimizer
    optimizer = torch.optim.SGD(ann.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4) if use_bn else \
        torch.optim.SGD(ann.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=num_epochs)

    # train
    for epoch in range(num_epochs):
        running_loss = 0
        start_time = time.time()
        ann.train()
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            labels = labels.to(device)
            images = images.to(device)
            outputs = ann(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            if (i + 1) % 40 == 0:
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.5f'
                      % (epoch + 1, num_epochs, i + 1, len(train_loader) // batch_size, running_loss))
                running_loss = 0
                print('Time elapsed:', time.time() - start_time)
        scheduler.step()
        correct = 0
        total = 0

        #test
        ann.eval()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = ann(inputs)
                loss = criterion(outputs, targets)
                _, predicted = outputs.cpu().max(1)
                total += float(targets.size(0))
                correct += float(predicted.eq(targets.cpu()).sum().item())
                if batch_idx % 100 == 0:
                    acc = 100. * float(correct) / float(total)
                    print(batch_idx, len(test_loader), ' Acc: %.5f' % acc)
        print('Iters:', epoch)
        print('Test Accuracy of the model on the 10000 test images: %.3f' % (100 * correct / total))
        acc = 100. * float(correct) / float(total)
        if best_acc < acc:
            best_acc = acc
            best_epoch = epoch + 1
            torch.save(ann.state_dict(), model_save_name,_use_new_zipfile_serialization=False)
        print('best_acc is: ', best_acc, ' find in epoch: ', best_epoch)
        print('\n\n')
