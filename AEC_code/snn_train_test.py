import torch
import random
import numpy as np
import os
import time
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100
from data.autoaugment import CIFAR10Policy, Cutout
from models.vgg import VGG
from models.new_convert_code_2 import SpikeModel,SpikeModule
import torch.nn.functional as F
from find_data_distribute import find_activation_percentile,GetLayerInputOutput
from models.fold_bn import search_fold_and_remove_bn
from data_base_Norm import data_norm_model
from models.utils import En_Decoding2

#Data set processing and loading

def build_data(dpath: str, batch_size=256, cutout=False,  use_cifar10=True, auto_aug=False):

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


batch_size = 128

epochs = 200
dataset = 'CIFAR10'
usebn = True



dpath = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(
    __file__)), '.','data_set'))

# spike number penalty term
def regularization(model:nn.Module,lamda):
   l1_loss = 0
   a=None
   for n,m in model.named_modules():
       if isinstance(m,nn.BatchNorm2d):
           a=m.bias.mean()
       if isinstance(m,En_Decoding2) and a is not None:
           b=torch.exp((-m.k * m.sim_length + m.t_d) ) * m.sita
           l1_loss+=(b-a)
           a=None

   return lamda*l1_loss
# spike number penalty term
def regularization1(model:nn.Module,lamda):
    l1_loss = 0
    for n, m in model.named_modules():
        if isinstance(m, En_Decoding2):
            a=(torch.exp((m.t_d_2) ) * m.sita_2-torch.exp((-m.k_2 * m.sim_length + m.t_d_2) ) * m.sita_2)
            l1_loss +=torch.norm(a)
    return lamda*l1_loss

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = epochs
    use_cifar10 = True
    train_loader, test_loader = build_data(dpath=dpath, cutout=True, use_cifar10=use_cifar10, auto_aug=True)
    best_acc = 0
    best_epoch = 0
    use_bn = usebn
    source_ann = VGG('VGG16', use_bn=use_bn, num_class=10 if use_cifar10 else 100)
    #Load the trained ANN model
    source_ann.load_state_dict(torch.load('ann.pth'))

    for name,p in source_ann.named_parameters():
        p.requires_grad = False


    snn = SpikeModel(model=source_ann, sim_length=5)

    for name,p in snn.named_modules():
        if isinstance(p,nn.BatchNorm2d):
            for n,m in p.named_parameters():
                m.requires_grad = True
    snn.to(device)


    criterion = nn.CrossEntropyLoss().to(device)


    optimizer =torch.optim.Adam(snn.parameters(),lr=0.00001,betas=(0.9,0.999),eps=1e-10,weight_decay=0)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=epochs)

    #train
    for epoch in range(num_epochs):
        running_loss = 0
        start_time = time.time()
        # ann.eval()
        # source_ann.train()
        snn.train()


        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            labels = labels.to(device)
            images = images.to(device)

            outputs = snn(images)
            loss = criterion(outputs, labels)

            #The loss function after adding the penalty term
            # loss =criterion(outputs, labels)+regularization1(snn,0.01)-regularization(snn,0.1)


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
        snn.eval()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = snn(inputs)
                _, predicted = outputs.cpu().max(1)
                total += float(targets.size(0))
                correct += float(predicted.eq(targets.cpu()).sum().item())
                if batch_idx % 10 == 0:
                    acc = 100. * float(correct) / float(total)
                    print(batch_idx, len(test_loader), ' Acc: %.5f' % acc)
                    # break
        print('Iters:', epoch)
        print('Test Accuracy of the model on the 10000 test images: %.3f' % (100 * correct / total))
        acc = 100. * float(correct) / float(total)
        if best_acc <= acc:
            best_acc = acc
            best_epoch = epoch + 1
            torch.save(snn.state_dict(), 'snn1015.pth',_use_new_zipfile_serialization=False)
        print('best_acc is: ', best_acc, ' find in epoch: ', best_epoch)

